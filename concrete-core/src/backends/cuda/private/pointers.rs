use std::ffi::c_void;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct StreamPointer(pub *mut c_void);

unsafe impl Sync for StreamPointer {}
unsafe impl Send for StreamPointer {}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct CPointer(pub *mut c_void);

unsafe impl Sync for CPointer {}
unsafe impl Send for CPointer {}
