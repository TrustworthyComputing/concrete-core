use crate::backends::cuda::engines::CudaError;
use crate::backends::cuda::implementation::engines::CudaEngine;
use crate::backends::cuda::implementation::entities::{
    CudaFourierLweBootstrapKey32, CudaFourierLweBootstrapKey64,
};
use crate::backends::cuda::private::crypto::bootstrap::{
    convert_lwe_bootstrap_key_from_cpu_to_gpu, CudaBootstrapKey,
};
use crate::prelude::{LweBootstrapKey32, LweBootstrapKey64};
use crate::specification::engines::{
    LweBootstrapKeyConversionGpuEngine, LweBootstrapKeyConversionGpuError,
};
use crate::specification::entities::LweBootstrapKeyEntity;
use std::marker::PhantomData;

impl From<CudaError> for LweBootstrapKeyConversionGpuError<CudaError> {
    fn from(err: CudaError) -> Self {
        Self::Engine(err)
    }
}

/// # Description
/// Convert an LWE bootstrap key corresponding to 32 bits of precision from the CPU to the GPU.
/// The bootstrap key is copied entirely to all the GPUs and converted from the standard to the
/// Fourier domain.

impl LweBootstrapKeyConversionGpuEngine<LweBootstrapKey32, CudaFourierLweBootstrapKey32>
    for CudaEngine
{
    /// # Example
    /// ```
    /// use concrete_core::backends::cuda::private::device::GpuIndex;
    /// use concrete_core::prelude::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    ///     Variance, *,
    /// };
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(1), PolynomialSize(512));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// const UNSAFE_SECRET: u128 = 0;
    /// let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    /// let lwe_sk: LweSecretKey32 = default_engine.generate_new_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey32 =
    ///     default_engine.generate_new_glwe_secret_key(glwe_dim, poly_size)?;
    /// let bsk: LweBootstrapKey32 =
    ///     default_engine.generate_new_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    ///
    /// let mut cuda_engine = CudaEngine::new(())?;
    /// let d_fourier_bsk: CudaFourierLweBootstrapKey32 =
    ///     cuda_engine.convert_lwe_bootstrap_key(&bsk)?;
    ///
    /// assert_eq!(d_fourier_bsk.glwe_dimension(), glwe_dim);
    /// assert_eq!(d_fourier_bsk.polynomial_size(), poly_size);
    /// assert_eq!(d_fourier_bsk.input_lwe_dimension(), lwe_dim);
    /// assert_eq!(d_fourier_bsk.decomposition_base_log(), dec_bl);
    /// assert_eq!(d_fourier_bsk.decomposition_level_count(), dec_lc);
    ///
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn convert_lwe_bootstrap_key(
        &self,
        input: &LweBootstrapKey32,
    ) -> Result<CudaFourierLweBootstrapKey32, LweBootstrapKeyConversionGpuError<CudaError>> {
        let poly_size = input.0.polynomial_size();
        check_poly_size!(poly_size);
        let data_per_gpu = input.glwe_dimension().to_glwe_size().0
            * input.glwe_dimension().to_glwe_size().0
            * input.input_lwe_dimension().0
            * input.decomposition_level_count().0
            * input.polynomial_size().0;
        let size = data_per_gpu as u64 * std::mem::size_of::<u32>() as u64;
        for stream in self.streams.iter() {
            stream.read().unwrap().check_device_memory(size)?;
        }
        Ok(unsafe { self.convert_lwe_bootstrap_key_unchecked(input) })
    }

    unsafe fn convert_lwe_bootstrap_key_unchecked(
        &self,
        input: &LweBootstrapKey32,
    ) -> CudaFourierLweBootstrapKey32 {
        let vecs = convert_lwe_bootstrap_key_from_cpu_to_gpu::<u32, _>(
            self.get_cuda_streams(),
            &input.0,
            self.get_number_of_gpus(),
        );
        CudaFourierLweBootstrapKey32(CudaBootstrapKey::<u32> {
            d_vecs: vecs,
            polynomial_size: input.polynomial_size(),
            input_lwe_dimension: input.input_lwe_dimension(),
            glwe_dimension: input.glwe_dimension(),
            decomp_level: input.decomposition_level_count(),
            decomp_base_log: input.decomposition_base_log(),
            _phantom: PhantomData::default(),
        })
    }
}

/// # Description
/// Convert an LWE bootstrap key corresponding to 64 bits of precision from the CPU to the GPU.
/// The bootstrap key is copied entirely to all the GPUs and converted from the standard to the
/// Fourier domain.

impl LweBootstrapKeyConversionGpuEngine<LweBootstrapKey64, CudaFourierLweBootstrapKey64>
    for CudaEngine
{
    /// # Example
    /// ```
    /// use concrete_core::backends::cuda::private::device::GpuIndex;
    /// use concrete_core::prelude::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    ///     Variance, *,
    /// };
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(1), PolynomialSize(512));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// const UNSAFE_SECRET: u128 = 0;
    /// let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    /// let lwe_sk: LweSecretKey64 = default_engine.generate_new_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey64 =
    ///     default_engine.generate_new_glwe_secret_key(glwe_dim, poly_size)?;
    /// let bsk: LweBootstrapKey64 =
    ///     default_engine.generate_new_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    ///
    /// let mut cuda_engine = CudaEngine::new(())?;
    /// let d_fourier_bsk: CudaFourierLweBootstrapKey64 =
    ///     cuda_engine.convert_lwe_bootstrap_key(&bsk)?;
    ///
    /// assert_eq!(d_fourier_bsk.glwe_dimension(), glwe_dim);
    /// assert_eq!(d_fourier_bsk.polynomial_size(), poly_size);
    /// assert_eq!(d_fourier_bsk.input_lwe_dimension(), lwe_dim);
    /// assert_eq!(d_fourier_bsk.decomposition_base_log(), dec_bl);
    /// assert_eq!(d_fourier_bsk.decomposition_level_count(), dec_lc);
    ///
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn convert_lwe_bootstrap_key(
        &self,
        input: &LweBootstrapKey64,
    ) -> Result<CudaFourierLweBootstrapKey64, LweBootstrapKeyConversionGpuError<CudaError>> {
        let poly_size = input.0.polynomial_size();
        check_poly_size!(poly_size);
        let data_per_gpu = input.glwe_dimension().to_glwe_size().0
            * input.glwe_dimension().to_glwe_size().0
            * input.input_lwe_dimension().0
            * input.decomposition_level_count().0
            * input.polynomial_size().0;
        let size = data_per_gpu as u64 * std::mem::size_of::<u64>() as u64;
        for stream in self.streams.iter() {
            stream.read().unwrap().check_device_memory(size)?;
        }
        Ok(unsafe { self.convert_lwe_bootstrap_key_unchecked(input) })
    }

    unsafe fn convert_lwe_bootstrap_key_unchecked(
        &self,
        input: &LweBootstrapKey64,
    ) -> CudaFourierLweBootstrapKey64 {
        let vecs = convert_lwe_bootstrap_key_from_cpu_to_gpu::<u64, _>(
            self.get_cuda_streams(),
            &input.0,
            self.get_number_of_gpus(),
        );
        CudaFourierLweBootstrapKey64(CudaBootstrapKey::<u64> {
            d_vecs: vecs,
            polynomial_size: input.polynomial_size(),
            input_lwe_dimension: input.input_lwe_dimension(),
            glwe_dimension: input.glwe_dimension(),
            decomp_level: input.decomposition_level_count(),
            decomp_base_log: input.decomposition_base_log(),
            _phantom: PhantomData::default(),
        })
    }
}
