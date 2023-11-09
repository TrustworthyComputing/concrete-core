use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LwePrivateFunctionalPackingKeyswitchKeyEntity;

engine_error! {
    LwePrivateFunctionalPackingKeyswitchKeyConversionGpuError for LwePrivateFunctionalPackingKeyswitchKeyConversionGpuEngine @
}

pub trait LwePrivateFunctionalPackingKeyswitchKeyConversionGpuEngine<Input, Output>:
    AbstractEngine
where
    Input: LwePrivateFunctionalPackingKeyswitchKeyEntity,
    Output: LwePrivateFunctionalPackingKeyswitchKeyEntity,
{
    fn convert_lwe_private_functional_packing_keyswitch_key(
        &self,
        input: &Input,
    ) -> Result<Output, LwePrivateFunctionalPackingKeyswitchKeyConversionGpuError<Self::EngineError>>;

    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LwePrivateFunctionalPackingKeyswitchKeyConversionError`]. For safety concerns
    /// _specific_ to an engine, refer to the implementer safety section.
    unsafe fn convert_lwe_private_functional_packing_keyswitch_key_unchecked(
        &self,
        input: &Input,
    ) -> Output;
}
