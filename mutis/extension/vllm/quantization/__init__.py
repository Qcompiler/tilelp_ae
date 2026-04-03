from mutis.utils.pkg import check_package_installed


 
# import mutis.extension.vllm.quantization.fp8_mutis
# import mutis.extension.vllm.quantization.mutis_quant
try:
    import mutis.extension.vllm.quantization.mutis_quant_new as mutis_quant
    from mutis.extension.vllm.quantization.mutis_quant_new import set_mutis_config
except ImportError:
    mutis_quant = None
    set_mutis_config = None
