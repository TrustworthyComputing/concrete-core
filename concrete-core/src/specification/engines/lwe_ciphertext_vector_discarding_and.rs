use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    LweBootstrapKeyEntity, LweCiphertextVectorEntity, LweKeyswitchKeyEntity,
};

engine_error! {
    LweCiphertextVectorDiscardingAndError for LweCiphertextVectorDiscardingAndEngine @
    LweDimensionMismatch => "The input and output LWE dimensions must be the same.",
    CiphertextCountMismatch => "The input and output ciphertext count must be the same."
}

impl<EngineError: std::error::Error> LweCiphertextVectorDiscardingAndError<EngineError> {
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
        if output.lwe_dimension() != input_1.lwe_dimension() {
            return Err(Self::LweDimensionMismatch);
        }
        if output.lwe_dimension() != input_2.lwe_dimension() {
            return Err(Self::LweDimensionMismatch);
        }
        if output.lwe_ciphertext_count() != input_1.lwe_ciphertext_count() {
            return Err(Self::CiphertextCountMismatch);
        }
        if output.lwe_ciphertext_count() != input_2.lwe_ciphertext_count() {
            return Err(Self::CiphertextCountMismatch);
        }
        Ok(())
    }
}

pub trait LweCiphertextVectorDiscardingAndEngine<
    InputCiphertextVector,
    OutputCiphertextVector,
    BootstrapKey,
    KeyswitchKey,
>: AbstractEngine where
    InputCiphertextVector: LweCiphertextVectorEntity,
    OutputCiphertextVector: LweCiphertextVectorEntity,
    BootstrapKey: LweBootstrapKeyEntity,
    KeyswitchKey: LweKeyswitchKeyEntity,
{
    fn discard_and_lwe_ciphertext_vector(
        &self,
        output: &mut OutputCiphertextVector,
        input_1: &InputCiphertextVector,
        input_2: &InputCiphertextVector,
        bsk: &BootstrapKey,
        ksk: &KeyswitchKey,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingAndError<Self::EngineError>>;

    unsafe fn discard_and_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut OutputCiphertextVector,
        input_1: &InputCiphertextVector,
        input_2: &InputCiphertextVector,
        bsk: &BootstrapKey,
        ksk: &KeyswitchKey,
        stream_idx: usize,
    );
}
