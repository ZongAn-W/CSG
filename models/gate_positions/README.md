# 动态＋地形 Gate 的四种位置实验

基于当前 RMSE 最优方案 `../ablations/04_dynamic_topography.py`。每个 `.py` 是独立上传模型，仅依赖 PyTorch；统一接口 `forward(x, topography)`，不需要 Ls 或臭氧通道索引。历史输入仍包含平台选定的全部变量。

原参照模型的 Gate 位于 SpatialEncoder 与 TemporalTranslator 之间。本目录四个模型均移走该 Gate，只在下述位置使用同一类乘法门控。

## 上传文件与实际位置

| 文件 | 平台模型名 | 路径中 Gate 的位置 | 门控动态输入 F |
|---|---|---|---|
| `01_before_convlstm.py` | CSG_GateBeforeConvLSTM | Input → Gate → ConvLSTM | 历史输入 `[B,T,C_in,H,W]` |
| `02_before_decoder.py` | CSG_GateBeforeDecoder | Translator → Gate → Decoder | 未来隐特征 `[B,K,S,h,w]` |
| `03_after_decoder.py` | CSG_GateAfterDecoder | Decoder → Gate → Output | 解码预测 `[B,K,1,H,W]` |
| `04_between_translator_blocks.py` | CSG_GateBetweenTranslatorBlocks | Block 1 → Gate 1 → Block 2 → … | 中间隐特征 `[B,D,h,w]` |

第四种严格定义为**相邻 block 之间**：N 个 block 有 N−1 个 Gate，最后一个 block 后不再加 Gate。默认2个block时是 `输入投影 → Block1 → Gate1 → Block2 → 输出投影`，只含1个Gate；3个block有2个Gate。该版本要求 `num_temporal_blocks >= 2`，各Gate独立拥有动态投影、地形编码器、输出投影和门强度。

Translator内部已经把历史时间并入通道，没有独立历史/未来时间维。为复用5维门控接口，第四种给其隐特征增加长度1的轴，门控后移除；它不是把整个历史窗口重复门控。

## 保持一致的计算

```
J = GELU(W_e(F) + f_topo(MOLA))
G = tanh(W_g(J))
F_out = F * (1 + s * G)
```

- G由当前位置的动态张量F和MOLA共同生成，不引入额外加性残差或FiLM。
- 地形编码器保持原来的两层卷积（首层stride=2）、高程除10000和dtype转换；按门控处空间大小双线性对齐。输入/输出门需要把地形编码插值回全分辨率。
- s初始1，仍可学习，沿用原始前向[0,1]直通截断；W_g保持Xavier gain=0.01。
- ③乘法系数在[0,2]，对解码器输出直接缩放，不能新增加性信号。若平台用标准化目标训练，缩放发生在标准化数值上，不能直接解释成物理浓度的倍数。
- ②③读取的是模型预测的隐表示/数值，不读取真实未来数据。

## 默认参数

| 参数 | 默认值 |
|---|---:|
| convlstm_hidden_dim | 16 |
| spatial_hidden_dim | 64 |
| temporal_hidden_dim | 128 |
| num_temporal_blocks | 2 |
| dropout | 0.1 |
| gate_hidden_dim | 32 |
| initial_gate_strength | 1.0 |

主配置仍由平台设置为历史20步、预测20步；也支持历史步与预测步不相等。

## 公平比较与参数量

各版本的主干模块与原④保持一致，但门控所在位置的通道数/分辨率不同，参数量和运算量会变化。第四种还随block数量增加门控数量，因此不是严格等参数实验。

在5个输入通道、20→20及上述默认值下，实测可训练参数量：

| 模型 | 参数量 |
|---|---:|
| 原④动态＋地形（Encoder后） | 542,914 |
| ①ConvLSTM前 | 539,079 |
| ②Decoder前 | 542,914 |
| ③Decoder后 | 538,819 |
| ④Translator block之间 | 547,074 |

新四个模型统一先构造主干再构造Gate，相同配置与随机种子下，公共主干初始权重相同，也与原无门控基线的主干一致。旧④在主干构造中间初始化Gate，因此不能声称旧④与新四组仅凭同seed就拥有相同初始权重；若要严格配对，应在平台复制同一份主干初始权重。

分别从头训练四组，并保留旧④作为原位置参照。统一实际输入通道、划分、归一化、优化器、学习率、batch size、训练预算和早停/checkpoint规则；以RMSE为主指标，报告MAE/SSIM及逐步误差。改变block数量也会改变主干容量，应保持各组block数一致。

## 验证

运行：
```
/opt/anaconda3/envs/ozone/bin/python -m unittest discover -s CSG/tests -p test_csg_gate_positions.py -v
```

专项测试覆盖：模板元数据及仅torch依赖、主干语法结构一致性、平台dry-run尺寸、20→20、历史/预测长度不同、多通道、奇数尺寸、float64地形、全分支梯度、门实际执行位置/次数、不同block深度、门算子与旧④在同权重下严格一致、动态/地形均影响门值、同seed公共主干一致、所有门强度归零后与基线逐元素等价。

测试通过只确认实现；尚未在平台训练这四种新模型，不能据此声称指标改善。
