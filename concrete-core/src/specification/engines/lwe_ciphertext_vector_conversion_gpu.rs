use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextVectorEntity;

engine_error! {
    LweCiphertextVectorConversionGpuError for LweCiphertextVectorConversionGpuEngine @
}

/// A trait for engines converting LWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a LWE ciphertext vector containing
/// the conversion of the `input` LWE ciphertext vector to a type with a different representation
/// (for instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait LweCiphertextVectorConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: LweCiphertextVectorEntity,
    Output: LweCiphertextVectorEntity,
{
    /// Converts a LWE ciphertext vector.
    fn convert_lwe_ciphertext_vector(
        &self,
        input: &Input,
    ) -> Result<Output, LweCiphertextVectorConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorConversionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn convert_lwe_ciphertext_vector_unchecked(&self, input: &Input) -> Output;
}
