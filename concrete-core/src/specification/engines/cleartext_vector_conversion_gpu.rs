use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::CleartextVectorEntity;

engine_error! {
    CleartextVectorConversionGpuError for CleartextVectorConversionGpuEngine @
}

/// A trait for engines converting (discard) cleartext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a cleartext vector containing the
/// conversion of the `input` cleartext vector to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait CleartextVectorConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: CleartextVectorEntity,
    Output: CleartextVectorEntity,
{
    /// Converts a cleartext vector.
    fn convert_cleartext_vector(
        &self,
        input: &Input,
    ) -> Result<Output, CleartextVectorConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a cleartext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`CleartextVectorConversionGpuError`]. For safety concerns _specific_ to an engine, refer
    /// to the implementer safety section.
    unsafe fn convert_cleartext_vector_unchecked(&self, input: &Input) -> Output;
}
