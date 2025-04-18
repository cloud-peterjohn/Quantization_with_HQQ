from hqq.core.quantize import BaseQuantizeConfig


# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    n_blocks = len(model.blocks)

    for i in range(n_blocks):
        quant_config[f"blocks.{i}.attn.qkv"] = BaseQuantizeConfig(
            nbits=4, axis=0, group_size=24
        )
        quant_config[f"blocks.{i}.attn.proj"] = BaseQuantizeConfig(
            nbits=4, axis=0, group_size=32
        )
        quant_config[f"blocks.{i}.mlp.fc1"] = BaseQuantizeConfig(
            nbits=4,
            group_size=48,
            axis=0,
        )
        quant_config[f"blocks.{i}.mlp.fc2"] = BaseQuantizeConfig(
            nbits=4,
            group_size=48,
            axis=0,
        )
        if i == 0 or i == len(model.blocks) - 1:  # first and last block
            quant_config[f"blocks.{i}.attn.qkv"] = BaseQuantizeConfig(
                nbits=4, axis=0, group_size=16
            )
            quant_config[f"blocks.{i}.attn.proj"] = BaseQuantizeConfig(
                nbits=4, axis=0, group_size=24
            )
            quant_config[f"blocks.{i}.mlp.fc1"] = BaseQuantizeConfig(
                nbits=4,
                group_size=32,
                axis=0,
            )
            quant_config[f"blocks.{i}.mlp.fc2"] = BaseQuantizeConfig(
                nbits=4,
                group_size=32,
                axis=0,
            )

    quant_config["head"] = BaseQuantizeConfig(
        nbits=6,
        group_size=24,
        axis=0,
    )

    return quant_config


# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}

    n_layers = model.config.num_hidden_layers

    for i in range(n_layers):
        if i <= n_layers // 4 or i >= (n_layers // 4) * 3:  # first and last 1/4
            quant_config[f"model.layers.{i}.self_attn.q_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
            quant_config[f"model.layers.{i}.self_attn.k_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
            quant_config[f"model.layers.{i}.self_attn.v_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
            quant_config[f"model.layers.{i}.self_attn.o_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
            quant_config[f"model.layers.{i}.mlp.gate_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=128
            )
            quant_config[f"model.layers.{i}.mlp.up_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=128
            )
            quant_config[f"model.layers.{i}.mlp.down_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=128
            )
        elif i == 0 or i == n_layers - 1:  # first and last layer
            quant_config[f"model.layers.{i}.self_attn.q_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=32
            )
            quant_config[f"model.layers.{i}.self_attn.k_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=32
            )
            quant_config[f"model.layers.{i}.self_attn.v_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=32
            )
            quant_config[f"model.layers.{i}.self_attn.o_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=32
            )
            quant_config[f"model.layers.{i}.mlp.gate_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
            quant_config[f"model.layers.{i}.mlp.up_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
            quant_config[f"model.layers.{i}.mlp.down_proj"] = BaseQuantizeConfig(
                nbits=8, group_size=64
            )
        else:  # middle layers
            # if i % 2 == 0:
            if True:
                quant_config[f"model.layers.{i}.self_attn.q_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=32
                )
                quant_config[f"model.layers.{i}.self_attn.k_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=32
                )
                quant_config[f"model.layers.{i}.self_attn.v_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=32
                )
                quant_config[f"model.layers.{i}.self_attn.o_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=32
                )
                quant_config[f"model.layers.{i}.mlp.gate_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=64
                )
                quant_config[f"model.layers.{i}.mlp.up_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=64
                )
                quant_config[f"model.layers.{i}.mlp.down_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=64
                )
            else:
                quant_config[f"model.layers.{i}.self_attn.q_proj"] = BaseQuantizeConfig(
                    nbits=8, group_size=128
                )
                quant_config[f"model.layers.{i}.self_attn.k_proj"] = BaseQuantizeConfig(
                    nbits=8, group_size=128
                )
                quant_config[f"model.layers.{i}.self_attn.v_proj"] = BaseQuantizeConfig(
                    nbits=8, group_size=128
                )
                quant_config[f"model.layers.{i}.self_attn.o_proj"] = BaseQuantizeConfig(
                    nbits=8, group_size=128
                )
                quant_config[f"model.layers.{i}.mlp.gate_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=64
                )
                quant_config[f"model.layers.{i}.mlp.up_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=64
                )
                quant_config[f"model.layers.{i}.mlp.down_proj"] = BaseQuantizeConfig(
                    nbits=4, group_size=64
                )

    return quant_config


# TODO: Make your own quant config for DeiT-S
# def get_quant_config_deit(model):
#     quant_config = {}

#     n_blocks = len(model.blocks)
#     q2_config = BaseQuantizeConfig(nbits=4, group_size=32)

#     for i in range(n_blocks):
#         quant_config[f'blocks.{i}.attn.qkv'] = q2_config
#         quant_config[f'blocks.{i}.attn.proj'] = q2_config
#         quant_config[f'blocks.{i}.mlp.fc1'] = q2_config
#         quant_config[f'blocks.{i}.mlp.fc2'] = q2_config

#     return quant_config


# TODO: Make your own quant config for Language Model
# def get_quant_config_slm(model):
#     quant_config = {}

#     n_layers = model.config.num_hidden_layers
#     q2_config = BaseQuantizeConfig(nbits=8, group_size=64)

#     for i in range(n_layers):
#         quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config
#         quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config
#         quant_config[f'model.layers.{i}.self_attn.v_proj'] = q2_config
#         quant_config[f'model.layers.{i}.self_attn.o_proj'] = q2_config

#         quant_config[f'model.layers.{i}.mlp.gate_proj'] = q2_config
#         quant_config[f'model.layers.{i}.mlp.up_proj'] = q2_config
#         quant_config[f'model.layers.{i}.mlp.down_proj'] = q2_config

#     return quant_config
