use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweKeyswitchKeyEntity;

engine_error! {
    LweKeyswitchKeyConversionGpuError for LweKeyswitchKeyConversionGpuEngine @
}

/// A trait for engines converting LWE keyswitch keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a LWE keyswitch key containing the
/// conversion of the `input` LWE keyswitch key to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait LweKeyswitchKeyConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: LweKeyswitchKeyEntity,
    Output: LweKeyswitchKeyEntity,
{
    /// Converts a LWE keyswitch key.
    fn convert_lwe_keyswitch_key(
        &self,
        input: &Input,
    ) -> Result<Output, LweKeyswitchKeyConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a LWE keyswitch key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweKeyswitchKeyConversionError`]. For safety concerns _specific_ to an engine, refer to
    /// the implementer safety section.
    unsafe fn convert_lwe_keyswitch_key_unchecked(&self, input: &Input) -> Output;
}
