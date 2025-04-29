import flashinfer
import torch
import pytest

@pytest.mark.parametrize("batch_size", [12, 61, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
def test_batch_prefill_with_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout
):
    torch.manual_seed(42)
    # variant for custom
    variant_decl = r"""
struct FlashTree : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t tree_len;
  uint32_t qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ FlashTree(const Params& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    tree_len = params.get_tree_len(batch_idx);
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);

    custom_mask_ptr = params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx];
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (kv_idx < tree_len) return true;
    const uint32_t col_idx = kv_idx - tree_len;
    const uint32_t offset = qo_idx * tree_len + col_idx;
    mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
    return mask;
  })
};
"""
    
    # use the provided arguments to create the JIT module
    jit_args = (
        "batch_prefill_flash_tree_mask",  # uri
        torch.bfloat16,  # dtype_q
        torch.bfloat16,  # dtype_kv
        torch.bfloat16,  # dtype_o
        torch.int32,  # idtype
        head_dim,  # hidden_dim_qk
        head_dim,  # hidden_dim_vo
        [],  # additional_tensor_names
        [],  # additional_tensor_dtypes
        [],  # additional_scalar_names
        [],  # additional_scalar_dtypes
        "FlashTree",
        variant_decl,
    )

    device = torch.device("cuda:0")

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout=kv_layout, backend="fa2", jit_args=jit_args
    )

    #* data
    # print(f"#Test: batch_size = {batch_size}, kv_len = {kv_len}, qo_len = {qo_len}, num_kv_heads = {num_kv_heads}, num_qo_heads = {num_qo_heads}")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else: #*
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
   
    kv_data = torch.randn(*kv_shape, dtype=torch.float16, device="cuda:0")
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    #* columns of mask < kv_len
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len // 2), True, device=device),
        diagonal=(kv_len // 2 - qo_len),
    ).reshape(-1)
    print(f"#Test: custom_mask size = {custom_mask.size()}")

    # use custom mask
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask
    )

    o_custom = wrapper.run(q, kv_data)
    
    print(o_custom.size())


if __name__ == "__main__":
    test_batch_prefill_with_custom_mask(61, 45, 23, 64, 64, 8, 128, "NHD")