//! Minimal DLPack producer for owned CPU bfloat16 arrays.
//!
//! The project intentionally does not depend on a DLPack crate.  The ABI is a
//! small, stable C interface and keeping the producer here lets us transfer an
//! ndarray allocation to PyTorch without first materialising a NumPy float32
//! array.  Ownership is carried by `DLManagedTensor::manager_ctx` and is
//! released by the managed-tensor deleter, as required by DLPack.

use half::bf16;
use ndarray::Array3;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

const DLPACK_NAME: &[u8] = b"dltensor\0";

// DLPack device and dtype constants.  These values are part of the public
// DLPack ABI (not PyTorch-specific values).
const K_DL_CPU: i32 = 1;
const K_DL_BFLOAT: u8 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Keeps every allocation referenced by a DLTensor alive.
///
/// `Array3` owns the bfloat16 storage.  Shape and stride arrays are kept in
/// boxed slices because DLPack stores raw pointers to them and requires them
/// to remain valid until the managed tensor is deleted.
struct DLPackOwner {
    // The field is intentionally only accessed by Drop: ownership of this
    // ndarray allocation is what keeps `data` valid for the consumer.
    #[allow(dead_code)]
    array: Array3<bf16>,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

unsafe extern "C" fn managed_tensor_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }

    // Reclaim the managed tensor itself first.  Its context owns the ndarray
    // allocation and the shape/stride metadata.
    let managed = Box::from_raw(managed);
    if !managed.manager_ctx.is_null() {
        drop(Box::from_raw(managed.manager_ctx.cast::<DLPackOwner>()));
    }
}

/// Python's capsule destructor must not call the managed deleter after a
/// consumer has taken ownership.  The DLPack convention is that consumers
/// rename the capsule from `dltensor` to `used_dltensor`; checking that name
/// here prevents a double free when the capsule is later garbage-collected.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    if capsule.is_null() {
        return;
    }

    let name = ffi::PyCapsule_GetName(capsule);
    if name.is_null() {
        // A Python exception raised from a destructor cannot be propagated.
        // Clear it so it does not leak into an unrelated Python operation.
        ffi::PyErr_Clear();
        return;
    }
    let name = CStr::from_ptr(name);
    if name.to_bytes() != &DLPACK_NAME[..DLPACK_NAME.len() - 1] {
        return;
    }

    let pointer = ffi::PyCapsule_GetPointer(capsule, DLPACK_NAME.as_ptr().cast::<c_char>());
    if pointer.is_null() {
        ffi::PyErr_Clear();
        return;
    }
    let managed = pointer.cast::<DLManagedTensor>();
    if let Some(deleter) = (*managed).deleter {
        deleter(managed);
    }
}

/// Turn an owned CPU bfloat16 ndarray into a DLPack capsule.
///
/// The returned capsule owns `array` until `torch.from_dlpack` consumes it and
/// invokes the managed-tensor deleter.  A non-standard-layout array is made
/// contiguous as a defensive fallback; the normal gdata read path already
/// returns standard-layout arrays, so that branch is not used for training.
pub(crate) fn into_dlpack<'py>(
    py: Python<'py>,
    mut array: Array3<bf16>,
) -> PyResult<Bound<'py, PyCapsule>> {
    if !array.is_standard_layout() {
        array = array.as_standard_layout().to_owned();
    }

    let shape: Box<[i64]> = array
        .shape()
        .iter()
        .map(|&dimension| dimension as i64)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let strides: Box<[i64]> = array
        .strides()
        .iter()
        .map(|&stride| stride as i64)
        .collect::<Vec<_>>()
        .into_boxed_slice();

    // `Array3` owns the allocation.  Its pointer remains stable when the
    // Array3 value is moved into DLPackOwner.
    let data = array.as_ptr() as *mut c_void;
    let owner = Box::new(DLPackOwner {
        array,
        shape,
        strides,
    });
    let owner = Box::into_raw(owner);
    let shape_ptr = unsafe { (*owner).shape.as_ptr() as *mut i64 };
    let strides_ptr = unsafe { (*owner).strides.as_ptr() as *mut i64 };

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device: DLDevice {
                device_type: K_DL_CPU,
                device_id: 0,
            },
            ndim: 3,
            dtype: DLDataType {
                code: K_DL_BFLOAT,
                bits: 16,
                lanes: 1,
            },
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        },
        manager_ctx: owner.cast::<c_void>(),
        deleter: Some(managed_tensor_deleter),
    });
    let managed = Box::into_raw(managed);

    let capsule = unsafe {
        ffi::PyCapsule_New(
            managed.cast::<c_void>(),
            DLPACK_NAME.as_ptr().cast::<c_char>(),
            Some(capsule_destructor),
        )
    };
    if capsule.is_null() {
        // PyCapsule_New does not take ownership when construction fails.
        unsafe { managed_tensor_deleter(managed) };
        return Err(PyErr::fetch(py));
    }

    // PyCapsule_New returns a new reference, which Bound takes ownership of.
    unsafe { Ok(Bound::from_owned_ptr_or_err(py, capsule)?.downcast_into()?) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn dtypes_match_dlpack_abi() {
        assert_eq!(std::mem::size_of::<bf16>(), 2);
        assert_eq!(std::mem::size_of::<DLDevice>(), 8);
        assert_eq!(std::mem::size_of::<DLDataType>(), 4);
        assert_eq!(
            Array3::<bf16>::from_elem((1, 2, 3), bf16::ZERO).strides(),
            &[6, 3, 1]
        );
    }
}
