use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextVectorEntity;

engine_error! {
    LweCiphertextVectorDiscardingAdditionGpuError for LweCiphertextVectorDiscardingAdditionGpuEngine @
    LweDimensionMismatch => "The input and output LWE dimensions must be the same.",
    CiphertextCountMismatch => "The input and output ciphertext count must be the same."
}

impl<EngineError: std::error::Error> LweCiphertextVectorDiscardingAdditionGpuError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks<OutputCiphertextVector, InputCiphertextVector>(
        output: &OutputCiphertextVector,
        input_1: &InputCiphertextVector,
        input_2: &InputCiphertextVector,
    ) -> Result<(), Self>
    where
        InputCiphertextVector: LweCiphertextVectorEntity,
        OutputCiphertextVector: LweCiphertextVectorEntity,
    {
        if output.lwe_dimension() != input_1.lwe_dimension()
            || output.lwe_dimension() != input_2.lwe_dimension()
        {
            return Err(Self::LweDimensionMismatch);
        }
        if output.lwe_ciphertext_count() != input_1.lwe_ciphertext_count()
            || output.lwe_ciphertext_count() != input_2.lwe_ciphertext_count()
        {
            return Err(Self::CiphertextCountMismatch);
        }
        Ok(())
    }
}

/// A trait for engines adding (discarding) LWE ciphertext vectors.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext vector
/// with the element-wise addition of the `input_1` LWE ciphertext vector and the `input_2` lwe
/// ciphertext vector.
///
/// # Formal Definition
///
/// cf [`here`](`crate::specification::engines::LweCiphertextDiscardingAdditionEngine`)
pub trait LweCiphertextVectorDiscardingAdditionGpuEngine<
    InputCiphertextVector,
    OutputCiphertextVector,
>: AbstractEngine where
    InputCiphertextVector: LweCiphertextVectorEntity,
    OutputCiphertextVector: LweCiphertextVectorEntity,
{
    /// Adds two LWE ciphertext vectors.
    fn discard_add_lwe_ciphertext_vector(
        &self,
        output: &mut OutputCiphertextVector,
        input_1: &InputCiphertextVector,
        input_2: &InputCiphertextVector,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingAdditionGpuError<Self::EngineError>>;

    /// Unsafely adds two LWE ciphertext vectors.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorDiscardingAdditionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_add_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut OutputCiphertextVector,
        input_1: &InputCiphertextVector,
        input_2: &InputCiphertextVector,
        stream_idx: usize,
    );
}
