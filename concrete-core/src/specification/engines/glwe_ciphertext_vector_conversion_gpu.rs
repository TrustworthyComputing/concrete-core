use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextVectorEntity;

engine_error! {
    GlweCiphertextVectorConversionGpuError for GlweCiphertextVectorConversionGpuEngine @
}

/// A trait for engines converting GLWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext vector containing
/// the conversion of the `input` GLWE ciphertext vector to a type with a different representation
/// (for instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait GlweCiphertextVectorConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: GlweCiphertextVectorEntity,
    Output: GlweCiphertextVectorEntity,
{
    /// Converts a GLWE ciphertext vector.
    fn convert_glwe_ciphertext_vector(
        &self,
        input: &Input,
    ) -> Result<Output, GlweCiphertextVectorConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a GLWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorConversionGpuError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn convert_glwe_ciphertext_vector_unchecked(&self, input: &Input) -> Output;
}
