#include "concrete-ffi.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

#include "utils.h"

void keyswitch_view_buffers_test(void) {
  // We generate the random sources
  DefaultEngine *engine = NULL;
  SeederBuilder *builder = get_best_seeder();

  int default_engine_ok = new_default_engine(builder, &engine);
  assert(default_engine_ok == 0);
  assert(engine != NULL);

  double ksk_variance = 0.000000000000000001;

  double encrypt_variance = 0.000000001;

  // We generate the key

  size_t input_lwe_dimension = 2;
  size_t output_lwe_dimension = 2;
  size_t level = 5;
  size_t base_log = 10;
  LweSecretKey64 *input_sk = NULL;
  int sk_ok = default_engine_create_lwe_secret_key_u64(engine, input_lwe_dimension, &input_sk);
  assert(sk_ok == 0);
  LweSecretKey64 *output_sk = NULL;
  sk_ok = default_engine_create_lwe_secret_key_u64(engine, input_lwe_dimension, &output_sk);
  assert(sk_ok == 0);

  LweKeyswitchKey64 *ksk = NULL;
  int ksk_ok = default_engine_create_lwe_keyswitch_key_u64(engine, input_sk, output_sk, level,
                                                           base_log, ksk_variance, &ksk);
  assert(ksk_ok == 0);

  // Test KSK Serialization/Deserialization
  Buffer ksk_buffer = {.pointer = NULL, .length = 0};
  int ksk_ser_ok = serialize_lwe_keyswitching_key_u64(ksk, &ksk_buffer);
  assert(ksk_ser_ok == 0);

  BufferView ksk_buffer_view = {.pointer = ksk_buffer.pointer, .length = ksk_buffer.length};
  LweKeyswitchKey64 *deser_ksk = NULL;
  int ksk_deser_ok = deserialize_lwe_keyswitching_key_u64(ksk_buffer_view, &deser_ksk);
  assert(ksk_deser_ok == 0);

  // We generate the texts
  uint64_t *input_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (input_lwe_dimension + 1));
  uint64_t *output_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (output_lwe_dimension + 1));
  uint64_t plaintext = ((uint64_t)1) << SHIFT;
  //   uint64_t plaintext = ((uint64_t)10) << SHIFT;

  LweCiphertextMutView64 *input_ct_as_mut_view = NULL;
  int input_ct_mut_view_ok = default_engine_create_lwe_ciphertext_mut_view_u64(
      engine, input_ct_buffer, input_lwe_dimension + 1, &input_ct_as_mut_view);
  assert(input_ct_mut_view_ok == 0);

  LweCiphertextView64 *input_ct_as_view = NULL;
  int input_ct_view_ok = default_engine_create_lwe_ciphertext_view_u64(
      engine, input_ct_buffer, input_lwe_dimension + 1, &input_ct_as_view);
  assert(input_ct_view_ok == 0);

  LweCiphertextMutView64 *output_ct_as_mut_view = NULL;
  int output_ct_mut_view_ok = default_engine_create_lwe_ciphertext_mut_view_u64(
      engine, output_ct_buffer, input_lwe_dimension + 1, &output_ct_as_mut_view);
  assert(output_ct_mut_view_ok == 0);

  LweCiphertextView64 *output_ct_as_view = NULL;
  int output_ct_view_ok = default_engine_create_lwe_ciphertext_view_u64(
      engine, output_ct_buffer, input_lwe_dimension + 1, &output_ct_as_view);
  assert(output_ct_view_ok == 0);

  // We encrypt the plaintext
  int encrypt_ok = default_engine_discard_encrypt_lwe_ciphertext_u64_view_buffers(
      engine, input_sk, input_ct_as_mut_view, plaintext, encrypt_variance);
  assert(encrypt_ok == 0);

  // We generate the keyswitch key and keyswitch
  int ks_ok = default_engine_discard_keyswitch_lwe_ciphertext_u64_view_buffers(
      engine, deser_ksk, output_ct_as_mut_view, input_ct_as_view);
  assert(ks_ok == 0);

  // We decrypt the plaintext
  uint64_t output = -1;
  int decrypt_ok = default_engine_decrypt_lwe_ciphertext_u64_view_buffers(
      engine, output_sk, output_ct_as_view, &output);
  assert(decrypt_ok == 0);

  // We check that the output are the same
  double expected = (double)plaintext / pow(2, SHIFT);
  double obtained = (double)output / pow(2, SHIFT);
  printf("Comparing output. Expected %f, Obtained %f\n", expected, obtained);
  double abs_diff = abs(obtained - expected);
  double rel_error = abs_diff / fmax(expected, obtained);
  assert(rel_error < 0.01);

  // We deallocate the objects
  default_engine_destroy_lwe_secret_key_u64(engine, input_sk);
  default_engine_destroy_lwe_secret_key_u64(engine, output_sk);
  default_engine_destroy_lwe_keyswitch_key_u64(engine, ksk);
  default_engine_destroy_lwe_keyswitch_key_u64(engine, deser_ksk);
  default_engine_destroy_lwe_ciphertext_view_u64(engine, input_ct_as_view);
  default_engine_destroy_lwe_ciphertext_mut_view_u64(engine, input_ct_as_mut_view);
  default_engine_destroy_lwe_ciphertext_view_u64(engine, output_ct_as_view);
  default_engine_destroy_lwe_ciphertext_mut_view_u64(engine, output_ct_as_mut_view);
  destroy_default_engine(engine);
  destroy_seeder_builder(builder);
  destroy_buffer(&ksk_buffer);
  free(input_ct_buffer);
  free(output_ct_buffer);
}

void keyswitch_unchecked_view_buffers_test(void) {
  // We generate the random sources
  DefaultEngine *engine = NULL;
  SeederBuilder *builder = get_best_seeder_unchecked();

  int default_engine_ok = new_default_engine_unchecked(builder, &engine);
  assert(default_engine_ok == 0);
  assert(engine != NULL);

  double ksk_variance = 0.000000000000000001;

  double encrypt_variance = 0.000000001;

  // We generate the key

  size_t input_lwe_dimension = 2;
  size_t output_lwe_dimension = 2;
  size_t level = 5;
  size_t base_log = 10;
  LweSecretKey64 *input_sk = NULL;
  int sk_ok =
      default_engine_create_lwe_secret_key_unchecked_u64(engine, input_lwe_dimension, &input_sk);
  assert(sk_ok == 0);
  LweSecretKey64 *output_sk = NULL;
  sk_ok =
      default_engine_create_lwe_secret_key_unchecked_u64(engine, input_lwe_dimension, &output_sk);
  assert(sk_ok == 0);

  LweKeyswitchKey64 *ksk = NULL;
  int ksk_ok = default_engine_create_lwe_keyswitch_key_unchecked_u64(
      engine, input_sk, output_sk, level, base_log, ksk_variance, &ksk);
  assert(ksk_ok == 0);

  // Test KSK Serialization/Deserialization
  Buffer ksk_buffer = {.pointer = NULL, .length = 0};
  int ksk_ser_ok = serialize_lwe_keyswitching_key_unchecked_u64(ksk, &ksk_buffer);
  assert(ksk_ser_ok == 0);

  BufferView ksk_buffer_view = {.pointer = ksk_buffer.pointer, .length = ksk_buffer.length};
  LweKeyswitchKey64 *deser_ksk = NULL;
  int ksk_deser_ok = deserialize_lwe_keyswitching_key_unchecked_u64(ksk_buffer_view, &deser_ksk);
  assert(ksk_deser_ok == 0);

  // We generate the texts
  uint64_t *input_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (input_lwe_dimension + 1));
  uint64_t *output_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (output_lwe_dimension + 1));
  uint64_t plaintext = ((uint64_t)1) << SHIFT;
  //   uint64_t plaintext = ((uint64_t)10) << SHIFT;

  LweCiphertextMutView64 *input_ct_as_mut_view = NULL;
  int input_ct_mut_view_ok = default_engine_create_lwe_ciphertext_mut_view_unchecked_u64(
      engine, input_ct_buffer, input_lwe_dimension + 1, &input_ct_as_mut_view);
  assert(input_ct_mut_view_ok == 0);

  LweCiphertextView64 *input_ct_as_view = NULL;
  int input_ct_view_ok = default_engine_create_lwe_ciphertext_view_unchecked_u64(
      engine, input_ct_buffer, input_lwe_dimension + 1, &input_ct_as_view);
  assert(input_ct_view_ok == 0);

  LweCiphertextMutView64 *output_ct_as_mut_view = NULL;
  int output_ct_mut_view_ok = default_engine_create_lwe_ciphertext_mut_view_unchecked_u64(
      engine, output_ct_buffer, input_lwe_dimension + 1, &output_ct_as_mut_view);
  assert(output_ct_mut_view_ok == 0);

  LweCiphertextView64 *output_ct_as_view = NULL;
  int output_ct_view_ok = default_engine_create_lwe_ciphertext_view_unchecked_u64(
      engine, output_ct_buffer, input_lwe_dimension + 1, &output_ct_as_view);
  assert(output_ct_view_ok == 0);

  // We encrypt the plaintext
  int encrypt_ok = default_engine_discard_encrypt_lwe_ciphertext_unchecked_u64_view_buffers(
      engine, input_sk, input_ct_as_mut_view, plaintext, encrypt_variance);
  assert(encrypt_ok == 0);

  // We generate the keyswitch key and keyswitch
  int ks_ok = default_engine_discard_keyswitch_lwe_ciphertext_unchecked_u64_view_buffers(
      engine, deser_ksk, output_ct_as_mut_view, input_ct_as_view);
  assert(ks_ok == 0);

  // We decrypt the plaintext
  uint64_t output = -1;
  int decrypt_ok = default_engine_decrypt_lwe_ciphertext_unchecked_u64_view_buffers(
      engine, output_sk, output_ct_as_view, &output);
  assert(decrypt_ok == 0);

  // We check that the output are the same
  double expected = (double)plaintext / pow(2, SHIFT);
  double obtained = (double)output / pow(2, SHIFT);
  printf("Comparing output. Expected %f, Obtained %f\n", expected, obtained);
  double abs_diff = abs(obtained - expected);
  double rel_error = abs_diff / fmax(expected, obtained);
  assert(rel_error < 0.01);

  // We deallocate the objects
  default_engine_destroy_lwe_secret_key_unchecked_u64(engine, input_sk);
  default_engine_destroy_lwe_secret_key_unchecked_u64(engine, output_sk);
  default_engine_destroy_lwe_keyswitch_key_unchecked_u64(engine, ksk);
  default_engine_destroy_lwe_keyswitch_key_unchecked_u64(engine, deser_ksk);
  default_engine_destroy_lwe_ciphertext_view_unchecked_u64(engine, input_ct_as_view);
  default_engine_destroy_lwe_ciphertext_mut_view_unchecked_u64(engine, input_ct_as_mut_view);
  default_engine_destroy_lwe_ciphertext_view_unchecked_u64(engine, output_ct_as_view);
  default_engine_destroy_lwe_ciphertext_mut_view_unchecked_u64(engine, output_ct_as_mut_view);
  destroy_default_engine_unchecked(engine);
  destroy_seeder_builder_unchecked(builder);
  destroy_buffer_unchecked(&ksk_buffer);
  free(input_ct_buffer);
  free(output_ct_buffer);
}

void keyswitch_raw_ptr_buffers_test(void) {
  // We generate the random sources
  DefaultEngine *engine = NULL;
  SeederBuilder *builder = get_best_seeder();

  int default_engine_ok = new_default_engine(builder, &engine);
  assert(default_engine_ok == 0);
  assert(engine != NULL);

  double ksk_variance = 0.000000000000000001;

  double encrypt_variance = 0.000000001;

  // We generate the key

  size_t input_lwe_dimension = 2;
  size_t output_lwe_dimension = 2;
  size_t level = 5;
  size_t base_log = 10;
  LweSecretKey64 *input_sk = NULL;
  int sk_ok = default_engine_create_lwe_secret_key_u64(engine, input_lwe_dimension, &input_sk);
  assert(sk_ok == 0);
  LweSecretKey64 *output_sk = NULL;
  sk_ok = default_engine_create_lwe_secret_key_u64(engine, input_lwe_dimension, &output_sk);
  assert(sk_ok == 0);

  LweKeyswitchKey64 *ksk = NULL;
  int ksk_ok = default_engine_create_lwe_keyswitch_key_u64(engine, input_sk, output_sk, level,
                                                           base_log, ksk_variance, &ksk);
  assert(ksk_ok == 0);

  // Test KSK Serialization/Deserialization
  Buffer ksk_buffer = {.pointer = NULL, .length = 0};
  int ksk_ser_ok = serialize_lwe_keyswitching_key_u64(ksk, &ksk_buffer);
  assert(ksk_ser_ok == 0);

  BufferView ksk_buffer_view = {.pointer = ksk_buffer.pointer, .length = ksk_buffer.length};
  LweKeyswitchKey64 *deser_ksk = NULL;
  int ksk_deser_ok = deserialize_lwe_keyswitching_key_u64(ksk_buffer_view, &deser_ksk);
  assert(ksk_deser_ok == 0);

  // We generate the texts
  uint64_t *input_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (input_lwe_dimension + 1));
  uint64_t *output_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (output_lwe_dimension + 1));
  uint64_t plaintext = ((uint64_t)1) << SHIFT;
  //   uint64_t plaintext = ((uint64_t)10) << SHIFT;

  // We encrypt the plaintext
  int encrypt_ok = default_engine_discard_encrypt_lwe_ciphertext_u64_raw_ptr_buffers(
      engine, input_sk, input_ct_buffer, plaintext, encrypt_variance);
  assert(encrypt_ok == 0);

  // We generate the keyswitch key and keyswitch
  int ks_ok = default_engine_discard_keyswitch_lwe_ciphertext_u64_raw_ptr_buffers(
      engine, deser_ksk, output_ct_buffer, input_ct_buffer);
  assert(ks_ok == 0);

  // We decrypt the plaintext
  uint64_t output = -1;
  int decrypt_ok = default_engine_decrypt_lwe_ciphertext_u64_raw_ptr_buffers(
      engine, output_sk, output_ct_buffer, &output);
  assert(decrypt_ok == 0);

  // We check that the output are the same
  double expected = (double)plaintext / pow(2, SHIFT);
  double obtained = (double)output / pow(2, SHIFT);
  printf("Comparing output. Expected %f, Obtained %f\n", expected, obtained);
  double abs_diff = abs(obtained - expected);
  double rel_error = abs_diff / fmax(expected, obtained);
  assert(rel_error < 0.01);

  // We deallocate the objects
  default_engine_destroy_lwe_secret_key_u64(engine, input_sk);
  default_engine_destroy_lwe_secret_key_u64(engine, output_sk);
  default_engine_destroy_lwe_keyswitch_key_u64(engine, ksk);
  default_engine_destroy_lwe_keyswitch_key_u64(engine, deser_ksk);
  destroy_default_engine(engine);
  destroy_seeder_builder(builder);
  destroy_buffer(&ksk_buffer);
  free(input_ct_buffer);
  free(output_ct_buffer);
}

void keyswitch_unchecked_raw_ptr_buffers_test(void) {
  // We generate the random sources
  DefaultEngine *engine = NULL;
  SeederBuilder *builder = get_best_seeder_unchecked();

  int default_engine_ok = new_default_engine_unchecked(builder, &engine);
  assert(default_engine_ok == 0);
  assert(engine != NULL);

  double ksk_variance = 0.000000000000000001;

  double encrypt_variance = 0.000000001;

  // We generate the key

  size_t input_lwe_dimension = 2;
  size_t output_lwe_dimension = 2;
  size_t level = 5;
  size_t base_log = 10;
  LweSecretKey64 *input_sk = NULL;
  int sk_ok =
      default_engine_create_lwe_secret_key_unchecked_u64(engine, input_lwe_dimension, &input_sk);
  assert(sk_ok == 0);
  LweSecretKey64 *output_sk = NULL;
  sk_ok =
      default_engine_create_lwe_secret_key_unchecked_u64(engine, input_lwe_dimension, &output_sk);
  assert(sk_ok == 0);

  LweKeyswitchKey64 *ksk = NULL;
  int ksk_ok = default_engine_create_lwe_keyswitch_key_unchecked_u64(
      engine, input_sk, output_sk, level, base_log, ksk_variance, &ksk);
  assert(ksk_ok == 0);

  // Test KSK Serialization/Deserialization
  Buffer ksk_buffer = {.pointer = NULL, .length = 0};
  int ksk_ser_ok = serialize_lwe_keyswitching_key_unchecked_u64(ksk, &ksk_buffer);
  assert(ksk_ser_ok == 0);

  BufferView ksk_buffer_view = {.pointer = ksk_buffer.pointer, .length = ksk_buffer.length};
  LweKeyswitchKey64 *deser_ksk = NULL;
  int ksk_deser_ok = deserialize_lwe_keyswitching_key_unchecked_u64(ksk_buffer_view, &deser_ksk);
  assert(ksk_deser_ok == 0);

  // We generate the texts
  uint64_t *input_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (input_lwe_dimension + 1));
  uint64_t *output_ct_buffer =
      aligned_alloc(U64_ALIGNMENT, sizeof(uint64_t) * (output_lwe_dimension + 1));
  uint64_t plaintext = ((uint64_t)1) << SHIFT;
  //   uint64_t plaintext = ((uint64_t)10) << SHIFT;

  // We encrypt the plaintext
  int encrypt_ok = default_engine_discard_encrypt_lwe_ciphertext_unchecked_u64_raw_ptr_buffers(
      engine, input_sk, input_ct_buffer, plaintext, encrypt_variance);
  assert(encrypt_ok == 0);

  // We generate the keyswitch key and keyswitch
  int ks_ok = default_engine_discard_keyswitch_lwe_ciphertext_unchecked_u64_raw_ptr_buffers(
      engine, deser_ksk, output_ct_buffer, input_ct_buffer);
  assert(ks_ok == 0);

  // We decrypt the plaintext
  uint64_t output = -1;
  int decrypt_ok = default_engine_decrypt_lwe_ciphertext_unchecked_u64_raw_ptr_buffers(
      engine, output_sk, output_ct_buffer, &output);
  assert(decrypt_ok == 0);

  // We check that the output are the same
  double expected = (double)plaintext / pow(2, SHIFT);
  double obtained = (double)output / pow(2, SHIFT);
  printf("Comparing output. Expected %f, Obtained %f\n", expected, obtained);
  double abs_diff = abs(obtained - expected);
  double rel_error = abs_diff / fmax(expected, obtained);
  assert(rel_error < 0.01);

  // We deallocate the objects
  default_engine_destroy_lwe_secret_key_unchecked_u64(engine, input_sk);
  default_engine_destroy_lwe_secret_key_unchecked_u64(engine, output_sk);
  default_engine_destroy_lwe_keyswitch_key_unchecked_u64(engine, ksk);
  default_engine_destroy_lwe_keyswitch_key_unchecked_u64(engine, deser_ksk);
  destroy_default_engine_unchecked(engine);
  destroy_seeder_builder_unchecked(builder);
  destroy_buffer_unchecked(&ksk_buffer);
  free(input_ct_buffer);
  free(output_ct_buffer);
}

int main(void) {
  keyswitch_view_buffers_test();
  keyswitch_unchecked_view_buffers_test();
  keyswitch_raw_ptr_buffers_test();
  keyswitch_unchecked_raw_ptr_buffers_test();
  return EXIT_SUCCESS;
}