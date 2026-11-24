# TopK / ArgMax / ArgMin 选择算子族 — 验证说明

## 交付内容

| 文件 | 作用 |
| --- | --- |
| `../op_topk.go` | 算子实现（不含任何外部依赖） |
| `../op_topk_test.go` | 26 个测试用例 |
| `verify_topk.sh` | 一键验证脚本（构建 + 新算子测试 + 全量回归） |
| `topk_verify.log` | 最近一次 `verify_topk.sh` 的完整输出 |

运行方式：

```bash
bash output/verify_topk.sh
```

脚本会设置该仓库基线就需要的 `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH`
兼容环境变量（仓库锁定的 `go4.org/unsafe/assume-no-moving-gc` 旧版本与新 Go
运行时的已知兼容问题，与本次改动无关），并依次执行：

1. `go build .`
2. `go test -count=1 -v -run 'TopK|ArgMax|ArgMin' .`
3. `go test -count=1 .`（整包回归，证明既有归约/求导/引擎行为不变）

## 算子结构

- `topKIndicesOp`：沿轴返回前 k 个最大（或最小）值的下标，k=1 即
  `ArgMax`/`ArgMin`。一元算子，输出固定为 `tensor.Int`，下标不参与元素
  类型推断。**不实现 SDOp/ADOp**，离散下标无梯度。
- `topKValuesOp`：二元算子 `(x, indices) -> values`，按下标聚集取值。
  实现 `SDOp`（符号微分）与 `ADOp`（自动微分），梯度仅回流到被选中的
  位置，同一位置被重复选中时累加。
- `topKValuesDiffOp`：反向散射算子。刻意以值方式内嵌 `topKBase` 而非
  `*topKValuesOp`，否则方法提升会让该符号梯度节点同时满足 ADOp，被
  lisp 机在自动求导阶段重复执行。
- 公共构图入口：`TopK(x, k, axis)`、`TopKMin(...)`、`ArgMax(x, axis)`、
  `ArgMin(x, axis)`。

## 语义约定（实现与测试共同锁定）

- 负轴按输入秩解释（与现有归约算子一致）：`-1` 表示最后一维。
- 并列值按原下标升序，`sort.SliceStable` 保证多次运行结果一致。
- NaN 排在所有非 NaN 值（含 ±Inf）之后；`+Inf`/`-Inf` 按普通序值参与。
- k=0 合法，输出保持秩、对应轴长度为 0；k 大于轴长时夹紧到轴长；
  空输入产出同秩空张量。
- 支持 `Float64`、`Float32`、`Int` 三种 dtype；整型输入的下标输出仍是
  `tensor.Int`，不参与类型推断；其他 dtype 返回明确错误。
- 输入数据永不修改（`OverwritesInput() == -1`），视图/转置输入先
  `Materialize` 到独立连续内存；输出全部新分配。
- 仅支持 CPU。设备内存上的张量带 `NativelyInaccessible` 标志，算子据此
  返回 `"... only implemented for CPU ..."` 的可识别错误，该判断在
  `!cuda` 与 `cuda` 两种构建下都成立。
- 哈希写入 axis/k/direction/d 及算子类型前缀，不同轴、不同 k、values 与
  indices 变体都不会被图缓存错误合并。

## 两种执行引擎

- tape（编译式）与 lisp（解释式）前向结果一致，见
  `TestTopKEngines_Agree`。
- 梯度：tape 机走 SDOp 符号微分（`TestTopKGrad_TapeMachine`，并与中心
  有限差分在 f64/f32 两种浮点下分别校验）；lisp 机走 ADOp `DoDiff`
  （`TestTopKGrad_LispMachine`，通过 `Backpropagate` 手动播种矩阵根的
  梯度）。
- 调查中确认 lisp 机对“归约 + 除法”链的 ADOp 路径存在**仓库既有**数值
  缺陷（纯 `Mean` 输入在 lisp 机下梯度即错误，与本算子无关），因此
  lisp 机的本算子测试隔离了上游链，直接验证本算子自身的反向契约。

## 失败与降级

- 轴越界、k 为负、秩不足：构图阶段（构造算子/`ApplyOp`）即返回带期望
  与实际值的上下文错误，见 `TestTopKOp_ConstructionErrors`、
  `TestTopKGraph_BadConstructionFailsAtBuild`。
- 对下标求导：链式两个不可微选择节点会让反向分析在构图阶段明确报
  `unable to differentiate ... would leave a portion of the graph
  unreachable`，而不是静默给零梯度，见
  `TestTopKIndices_NotDifferentiable`。

## 环境备注

`go build -tags cuda ./...` 在本机失败，原因是缺少 CUDA 工具链
（`gorgonia.org/cu/blas` 编译不过）；在未改动的基线提交上行为完全相同，
故与本次改动无关。
