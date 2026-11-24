# 求导生命周期、梯度缓冲与梯度校验 — 验证说明

本目录记录本次改动：为同一张计算图上的重复求导提供生命周期、为梯度缓冲提供复用与可观测语义、
为梯度提供数值校验，并让训练循环的梯度在每个迭代起点保持干净。

## 交付文件

| 文件 | 作用 |
| --- | --- |
| `differentiation_lifecycle.go` | 求导生命周期：`Differentiate`、`Grad`、槽位/驱逐/回滚、`DifferentiationError` 与错误种类、`ResetDifferentiation`、状态查询 |
| `differentiation.go`（改） | 抽出 `backpropagate(..., prune bool)`；公开 `Backpropagate` 行为不变 |
| `gorgonia.go`（改） | `Grad` 改为 `Differentiate` 的薄封装，签名与调用方式不变 |
| `graph.go`（改） | `ExprGraph` 增加求导账本字段（互斥量、槽位、基线快照） |
| `node.go` / `perf.go`（改） | `Node.gradDirty` 标记与节点池复位 |
| `gradient_pool.go` | 按形状与类型分组的梯度缓冲池；取出即归零；`GradBuffer` 生命周期；`GradientPoolStats` |
| `grad_check.go` | 数值差分梯度校验；`GradCheck` / `GradCheckWithConfig` / `GradCheckStrict*`；`GradCheckReport` / `GradCheckError` |
| `solver_lifecycle.go` | `MarkGradDirty` / `ZeroGrad` / `SolverStepSafe` / `TrainingSession` / `ZeroNodeGradient` |
| `differentiation_lifecycle_test.go` | 17 个用例：重复求导、共用子图、节点计数、重置、错误种类、回滚、并发、陈旧标记、损坏状态 |
| `gradient_pool_test.go` | 11 个用例：复用、归零、分组、失效句柄、独占、统计、并发、拒绝竞争、释放 |
| `grad_check_test.go` | 16 个用例（含逐算子子测试；共 53 个新用例）：标量/矩阵/全连接/逐元素/复合表达式、确定性、坏导数注入、跳过、容差、f32 |
| `solver_lifecycle_test.go` | 9 个用例：每轮干净、失败后干净、与独立运行一致、缓冲借还、异常路径归还、并发会话、全循环收敛 |
| `verify_lifecycle.sh` | 一键验证脚本（构建 + 新用例 + 全量回归 + 竞态检测） |
| `lifecycle_verify.log` | 最近一次运行的完整输出 |

运行方式：

```bash
bash output/verify_lifecycle.sh
```

脚本会设置仓库基线就需要的 `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH` 兼容环境变量
（仓库锁定的 `go4.org/unsafe/assume-no-moving-gc` 旧版本与较新 Go 运行时的已知兼容问题，
与本次改动无关），并依次执行：

1. `go build .`
2. `go test -count=1 -v -run 'TestDifferentiate|TestGradCheck|TestGradientPool|TestTrainingSession|TestMarkGradDirty' .`
3. `go test -count=1 -race -run '<同一组>' .`
4. `go test -count=1 .`（整包回归，证明既有求导/求解器/执行机/图行为不变）

## 一、求导生命周期

历史行为：`Backpropagate` 把每个节点的导数写进 `Node.deriv` 后从不失效。于是同一张图上第二次
`Grad` 直接复用第一次的标记（返回旧结果）、对共用子图的另一个代价求导会报“没有梯度”、
每次求导都在图里再挂一套导数节点。

现在的语义：

- **同图可反复求导**：同一 `(cost, 参数集合)` 重复求导是幂等的 —— 命中槽位缓存，返回同一批
  导数节点，图的节点/边数完全不变（`TestDifferentiate_RepeatSameCostStable`、
  `TestDifferentiate_DeterministicAcrossRuns`）。
- **共用子图**：先后对共享子图的两个代价求导，第二条路径先驱逐上一条，再从干净状态重建，
  结果与在全新图上重建逐元素相等（`TestDifferentiate_SharedSubgraphMatchesFreshGraph`、
  `TestDifferentiate_MatrixSharedSubgraph`）。
- **有界增长**：反复在多个代价之间切换，节点数在单次求导的规模处封顶，不随次数增长
  （`TestDifferentiate_NodeCountBounded`、`TestTrainingSession_FullLoopConverges`）。
- **不误伤**：`Differentiate` 传入 `prune=false`，不会像旧路径那样顺手删掉孤立用户节点。

槽位/驱逐规则（`differentiation_lifecycle.go` 顶部注释）：

- 求导成功即记录一个“槽位”，含它创建的导数节点、之后的 `deriv`/`derivOf` 快照、以及依赖足迹。
- 足迹互不相交的槽位可以共存 —— 这正是仓库既有 `TestMisha` 多输出模型依赖的形态。
- 新请求的足迹或参数集合与任一槽位相交，则先驱逐全部槽位：移除它们创建的节点、恢复
  `deriv`/`derivOf` 到求导前状态，再重新求导。因此结果等价于全新图，节点数不累积。
- 重复同一请求走缓存，完全不动图。

`ResetDifferentiation()` 显式回到干净状态：移除全部求导产物、恢复标记与根集合、清空损坏标记。
已有节点的值、类型、连接不被修改（`TestDifferentiate_ResetRestoresGraph`、
`TestDifferentiate_RepeatedResetCycle`）。

## 二、失败与降级

`DifferentiationError` 携带可判定的 `Kind`，绝不退化成 “no gradient”：

| Kind | 场景 |
| --- | --- |
| `DiffErrCostNil` | 代价为空 |
| `DiffErrCostNotScalar` | 代价不是标量 |
| `DiffErrCostNotInGraph` | 代价不属于任何图 |
| `DiffErrWRTNil` | 参数为空 |
| `DiffErrWRTNotInput` | 参数不是输入节点 |
| `DiffErrWRTNotInGraph` | 参数不属于代价所在的图 |
| `DiffErrUnsupportedDtype` | 非浮点或未支持的代价类型 |
| `DiffErrNotDifferentiable` | 符号微分本身失败 |
| `DiffErrStaleMarks` | 图上存在非本生命周期创建的导数标记（例如直接调用过 `Backpropagate`） |
| `DiffErrGraphMarked` | 图已被标记为不可再求导，附带原因 |

- 失败即回滚：在求导中途失败会把新增节点移除、标记恢复、根集合还原，图要么回到可再次求导的
  干净状态，要么被显式标记为 `GradGraphBroken` 并给出去不掉的原因
  （`TestDifferentiate_FailureRollsBack`、`TestDifferentiate_RecoversAfterRollbackFailure`、
  `TestDifferentiate_BrokenGraphIsExplicit`）。
- 陈旧标记不会被静默消费：返回 `DiffErrStaleMarks` 并提示先调用 `ResetDifferentiation` 或
  `ClearDerivativeMarks` 收养该图（`TestDifferentiate_StaleMarksRejected`、
  `TestDifferentiate_BackpropagateStillWorks`）。

## 三、梯度缓冲池

`GradientPool` 按 `(形状, 类型)` 分组复用缓冲，并提供只含数值、不持有缓冲引用的统计快照。

- **取出即归零**：`Get` 在把缓冲交给调用者的那一刻清零，绝不在使用后清零（否则会抹掉本轮刚
  累加的项）。`Stats().ZerosOnBorrow` 计数（`TestGradientPool_ReuseAndZeroOnBorrow`、
  `TestTrainingSession_GradientBufferZeroAtTakeOut`）。
- **唯一权威账本**：每个槽位带世代号；`Put` 后旧句柄永久失效，再读返回 `ErrBufferReleased`，
  不会崩溃也不会读到旧值。同一存储可被重新发出给新句柄，旧句柄仍失效
  （`TestGradientPool_ReleasedHandleIsInvalid`、`TestGradientPool_StaleHandleAfterReissue`）。
- **不共享**：两个在借缓冲永不指向同一存储（`TestGradientPool_ExclusiveOwnership`）。
- **复用边界**：`Retain`/`Release` 引用计数；被执行机或求解器缓存持有期间 `Put` 返回
  `ErrBufferPinned`，归还后才可回收（`TestGradientPool_RetainBlocksReturn`）。
- **并发**：同一池内的取用默认串行化；`WithNonBlockingTakeOut` 时并发取用明确返回
  `ErrPoolBusy` 而非排队（`TestGradientPool_ConcurrentUse`、`TestGradientPool_TryGetRejectsWhenBusy`）。
- **观测**：新建、命中、取出归零、显式归零、归还、在借、空闲、峰值在借、字节数、峰值字节数、
  分组数均可从 `Stats()` 取到，并发下不丢（`TestGradientPool_StatsSnapshotHoldsNoBuffers`）。
- 仅接受 `Float64`/`Float32`；其它类型返回 `ErrBufferNotFloat`。

## 四、梯度校验

`GradCheck` 对给定代价与参数，用中心差分逐元素比较解析梯度：

- 报告含逐参数绝对误差、相对误差、最大绝对/相对误差的位置（参数名 + 下标）与结论
  （`GradCheckReport.String()`），阈值可配（`GradCheckConfig`）。
- 可复现：扰动方向固定为 `+e_i` / `-e_i`，按行主序，报告按参数顺序、下标顺序输出；同一图、
  同一取值下两次报告逐字一致（`TestGradCheck_Deterministic`）。校验结束后参数值与图的导数
  绑定都恢复到校验前。
- 失败即失败：`GradCheckStrict*` 在未通过时返回 `*GradCheckError`，指出最差参数、误差量级与阈值，
  不只是在日志里打一行（`TestGradCheck_WrongGradientPointsAtParameter` 用一个故意丢掉因子 2
  的算子证明会判失败并指到 `x`）。
- 跳过需记账：非浮点、形状不符、无值、不可微、结果非有限都记入 `Skipped` 并给出原因与计数
  （`TestGradCheck_SkipsNonFloat` 等）。

解析梯度覆盖：标量、矩阵乘、逐元素（exp/ln/tanh/sigmoid/cube/sqrt/inverse/sin/cos/softplus）、
归约与复合表达式，均与有限差分一致（`TestGradCheck_*`）。

## 五、训练循环集成

- `TrainingSession.BeginIteration` 在每轮起点归零全部参数梯度、清除脏标记；这是权威的归零点，
  **不在**求解器用完后再清（`TestTrainingSession_CleanGradientEachIteration`）。
- `SolverStepSafe` 在求解器中途失败时把模型参数的梯度归零并标脏，返回 `*ErrSolverStep`；
  下一轮从零开始，不累加残值（`TestTrainingSession_GradientCleanAfterFailure`）。
- 每轮梯度与“在相同参数点上全新构建的图独立计算”逐元素一致
  （`TestTrainingSession_IterationMatchesIndependentRun`）。
- 缓冲借还在一轮内成对进行，异常路径经 `defer EndIteration` 也能归还，不泄漏
  （`TestTrainingSession_BufferLifecycle`、`TestTrainingSession_BuffersReturnedOnFailure`）。
- 完整循环收敛且节点数在首轮达峰后不再增长（`TestTrainingSession_FullLoopConverges`）。
- 多个会话并发跑一遍，`-race` 下全部通过（`TestTrainingSession_ConcurrentSessions`、
  `TestDifferentiate_ConcurrentSessions`）。

## 六、未破坏的既有行为

- 公开约定不变：`Grad(cost, wrt...)` 的签名与调用方式不变，`Backpropagate` 的语义（包括修剪
  孤立节点）不变；现有训练代码无需改动。
- 单次求导数值结果、求解器原地更新（顺序、批缩放、正则与裁剪先后、历史量保留）、执行机的
  绑定/寄存器/重置/关闭语义、图复制/删点/回收/设备搬运，均由整包既有用例覆盖并全绿。
- 执行机保持“重复 `RunAll` 累加梯度”的历史语义不变；清零由 `TrainingSession` 在每轮起点显式
  完成，避免改变既有行为。
- 无新增第三方依赖；纯本地路径成立。`cuda` 构建标签在本机因缺少 CUDA 工具链无法完整编译，
  该行为在未改动的基线提交上完全相同（缺 `cuda.h`/`cudnn.h`），与本次改动无关。
