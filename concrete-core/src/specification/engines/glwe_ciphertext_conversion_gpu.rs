use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextConversionGpuError for GlweCiphertextConversionGpuEngine @
}

/// A trait for engines converting GLWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext containing the
/// conversion of the `input` GLWE ciphertext to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait GlweCiphertextConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: GlweCiphertextEntity,
    Output: GlweCiphertextEntity,
{
    /// Converts a GLWE ciphertext.
    fn convert_glwe_ciphertext(
        &self,
        input: &Input,
    ) -> Result<Output, GlweCiphertextConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a GLWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextConversionGpuError`]. For safety concerns _specific_ to an engine, refer
    /// to the implementer safety section.
    unsafe fn convert_glwe_ciphertext_unchecked(&self, input: &Input) -> Output;
}
