use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextDiscardingConversionGpuError for GlweCiphertextDiscardingConversionGpuEngine @
    GlweDimensionMismatch => "The input and output GLWE dimension must be the same.",
    PolynomialSizeMismatch => "The input and output polynomial size must be the same."
}

impl<EngineError: std::error::Error> GlweCiphertextDiscardingConversionGpuError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks<Input, Output>(output: &Output, input: &Input) -> Result<(), Self>
    where
        Input: GlweCiphertextEntity,
        Output: GlweCiphertextEntity,
    {
        if input.glwe_dimension() != output.glwe_dimension() {
            return Err(Self::GlweDimensionMismatch);
        }

        if input.polynomial_size() != output.polynomial_size() {
            return Err(Self::PolynomialSizeMismatch);
        }

        Ok(())
    }
}

/// A trait for engines converting (discarding) GLWE ciphertexts .
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE ciphertext with
/// the conversion of the `input` GLWE ciphertext to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait GlweCiphertextDiscardingConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: GlweCiphertextEntity,
    Output: GlweCiphertextEntity,
{
    /// Converts a GLWE ciphertext .
    fn discard_convert_glwe_ciphertext(
        &self,
        output: &mut Output,
        input: &Input,
    ) -> Result<(), GlweCiphertextDiscardingConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a GLWE ciphertext .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextDiscardingConversionGpuError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_convert_glwe_ciphertext_unchecked(&self, output: &mut Output, input: &Input);
}
