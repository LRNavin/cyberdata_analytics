       �K"	  @�B=�Abrain.Event:2� ��     ����	2pv�B=�A"��
j
input_1Placeholder*
dtype0*'
_output_shapes
:���������+*
shape:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+   @   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�{r�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�{r>
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*

seed*
T0*
dtype0*
_output_shapes

:+@*
seed2끞
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:+@
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+@
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:+@*
	container *
shape
:+@
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+@*
use_locking(
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+@
Z
dense_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_1/bias
VariableV2*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:@
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������@*
transpose_a( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:���������@*
T0*
data_formatNHWC
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������@
g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������@
g
"dense_1/activity_regularizer/mul/xConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/mulMul"dense_1/activity_regularizer/mul/x dense_1/activity_regularizer/Abs*
T0*'
_output_shapes
:���������@
s
"dense_1/activity_regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
g
"dense_1/activity_regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/addAdd"dense_1/activity_regularizer/add/x dense_1/activity_regularizer/Sum*
T0*
_output_shapes
: 
m
dense_2/random_uniform/shapeConst*
valueB"@       *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *  ��*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *  �>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*

seed*
T0*
dtype0*
_output_shapes

:@ *
seed2�ߜ
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:@ 
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:@ 
�
dense_2/kernel
VariableV2*
dtype0*
_output_shapes

:@ *
	container *
shape
:@ *
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:@ 
{
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes

:@ *
T0*!
_class
loc:@dense_2/kernel
Z
dense_2/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
: 
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
: 
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
T0*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:��������� *
T0
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:��������� 
m
dense_3/random_uniform/shapeConst*
valueB"        *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *qĜ>*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes

:  *
seed2��*

seed*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:  
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:  
�
dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:  *
	container *
shape
:  *
shared_name 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:  
{
dense_3/kernel/readIdentitydense_3/kernel*
_output_shapes

:  *
T0*!
_class
loc:@dense_3/kernel
Z
dense_3/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
x
dense_3/bias
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
: 
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
T0*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*'
_output_shapes
:��������� *
T0*
data_formatNHWC
W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:��������� 
m
dense_4/random_uniform/shapeConst*
valueB"    +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *�А�*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *�А>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
_output_shapes

: +*
seed2�Ֆ*

seed*
T0*
dtype0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

: +
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

: +
�
dense_4/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

: +*
	container *
shape
: +
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

: +
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

: +
Z
dense_4/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
x
dense_4/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:+*
	container *
shape:+
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:+
�
dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*
T0*'
_output_shapes
:���������+*
transpose_a( *
transpose_b( 
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������+
W
dense_4/ReluReludense_4/BiasAdd*
T0*'
_output_shapes
:���������+
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
_output_shapes
: *
T0
�
dense_4_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
q
dense_4_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
l
loss/dense_4_loss/subSubdense_4/Reludense_4_target*
T0*'
_output_shapes
:���������+
k
loss/dense_4_loss/SquareSquareloss/dense_4_loss/sub*'
_output_shapes
:���������+*
T0
s
(loss/dense_4_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
m
*loss/dense_4_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Mean*loss/dense_4_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
|
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*#
_output_shapes
:���������*
T0
a
loss/dense_4_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*#
_output_shapes
:���������*
T0
�
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
a
loss/dense_4_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_2*
T0*#
_output_shapes
:���������
c
loss/dense_4_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
loss/dense_4_loss/Mean_3Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_4_loss/Mean_3*
T0*
_output_shapes
: 
\
loss/addAddloss/mul dense_1/activity_regularizer/add*
T0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
metrics/acc/ArgMaxArgMaxdense_4_targetmetrics/acc/ArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:���������
x
metrics/acc/CastCastmetrics/acc/Equal*#
_output_shapes
:���������*

DstT0*

SrcT0
*
Truncate( 
[
metrics/acc/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
}
training/Adam/gradients/ShapeConst*
valueB *
_class
loc:@loss/add*
dtype0*
_output_shapes
: 
�
!training/Adam/gradients/grad_ys_0Const*
valueB
 *  �?*
_class
loc:@loss/add*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*

index_type0*
_class
loc:@loss/add*
_output_shapes
: 
�
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_3*
_output_shapes
: *
T0*
_class
loc:@loss/mul
�
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1Shapeloss/dense_4_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2Const*
valueB *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ConstConst*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
_output_shapes

:*
T0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
T0*
out_type0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
_output_shapes
:
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*'
_output_shapes
:���������@*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
out_type0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*
T0
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
valueB *,
_class"
 loc:@loss/dense_4_loss/truediv*
dtype0*
_output_shapes
: 
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:���������:���������
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeConst*
valueB *3
_class)
'%loc:@dense_1/activity_regularizer/mul*
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1Shape dense_1/activity_regularizer/Abs*
_output_shapes
:*
T0*
out_type0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Straining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeEtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*2
_output_shapes 
:���������:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*'
_output_shapes
:���������@*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������@*
T0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Utraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:
�
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������@
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������*
T0
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/mul
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/mul
�
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������@
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������@
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*
_output_shapes
:*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeConst*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/addAdd*loss/dense_4_loss/Mean_1/reduction_indices:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *+
_class!
loc:@loss/dense_4_loss/Mean_1
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:*

Tidx0
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/valueConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*

index_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
T0
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
N*
_output_shapes
:
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/MaximumMaximumCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordivFloorDiv;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*
T0*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Reshape>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������*

Tmultiples0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2Shapeloss/dense_4_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
T0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/CastCast@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeloss/dense_4_loss/Square*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *)
_class
loc:@loss/dense_4_loss/Mean
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
value	B : *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*
_output_shapes
:*

Tidx0*)
_class
loc:@loss/dense_4_loss/Mean
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
T0*

index_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*)
_class
loc:@loss/dense_4_loss/Mean*
N*
_output_shapes
:*
T0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2Shapeloss/dense_4_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
:
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_4_loss/Mean*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/ConstConst<^training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv*
_output_shapes
: *
valueB
 *   @*+
_class!
loc:@loss/dense_4_loss/Square*
dtype0
�
9training/Adam/gradients/loss/dense_4_loss/Square_grad/MulMulloss/dense_4_loss/sub;training/Adam/gradients/loss/dense_4_loss/Square_grad/Const*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Mul;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv9training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/ShapeShapedense_4/Relu*
_output_shapes
:*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*(
_class
loc:@loss/dense_4_loss/sub
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/SumSum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/sub
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/sub*'
_output_shapes
:���������+
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*0
_output_shapes
:������������������*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/sub
�
2training/Adam/gradients/dense_4/Relu_grad/ReluGradReluGrad:training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshapedense_4/Relu*'
_output_shapes
:���������+*
T0*
_class
loc:@dense_4/Relu
�
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:+*
T0*"
_class
loc:@dense_4/BiasAdd
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:��������� *
transpose_a( 
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*!
_class
loc:@dense_4/MatMul*
_output_shapes

: +*
transpose_a(*
transpose_b( *
T0
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*
T0*
_class
loc:@dense_3/Tanh*'
_output_shapes
:��������� 
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
: 
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:��������� *
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
_output_shapes

:  *
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_3/MatMul
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:��������� 
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
: 
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:@ *
transpose_a(
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������@
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������@
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:@
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:���������+*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul*
_output_shapes

:+@*
transpose_a(
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
p
training/Adam/CastCastAdam/iterations/read*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
t
#training/Adam/zeros/shape_as_tensorConst*
valueB"+   @   *
dtype0*
_output_shapes
:
^
training/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
T0*

index_type0*
_output_shapes

:+@
�
training/Adam/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:+@*
	container *
shape
:+@
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
_output_shapes

:+@*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+@*
T0
b
training/Adam/zeros_1Const*
dtype0*
_output_shapes
:@*
valueB@*    
�
training/Adam/Variable_1
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_1
v
%training/Adam/zeros_2/shape_as_tensorConst*
valueB"@       *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*

index_type0*
_output_shapes

:@ *
T0
�
training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes

:@ *
	container *
shape
:@ 
�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
validate_shape(*
_output_shapes

:@ *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2
�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:@ *
T0
b
training/Adam/zeros_3Const*
valueB *    *
dtype0*
_output_shapes
: 
�
training/Adam/Variable_3
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: *
use_locking(
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
: *
T0
v
%training/Adam/zeros_4/shape_as_tensorConst*
valueB"        *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*
_output_shapes

:  
�
training/Adam/Variable_4
VariableV2*
dtype0*
_output_shapes

:  *
	container *
shape
:  *
shared_name 
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:  *
use_locking(*
T0
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:  
b
training/Adam/zeros_5Const*
dtype0*
_output_shapes
: *
valueB *    
�
training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
: *
use_locking(
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
: 
v
%training/Adam/zeros_6/shape_as_tensorConst*
valueB"    +   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*
_output_shapes

: +
�
training/Adam/Variable_6
VariableV2*
dtype0*
_output_shapes

: +*
	container *
shape
: +*
shared_name 
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
validate_shape(*
_output_shapes

: +*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

: +
b
training/Adam/zeros_7Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_7
VariableV2*
shared_name *
dtype0*
_output_shapes
:+*
	container *
shape:+
�
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:+
�
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:+
v
%training/Adam/zeros_8/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"+   @   
`
training/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*
_output_shapes

:+@
�
training/Adam/Variable_8
VariableV2*
shape
:+@*
shared_name *
dtype0*
_output_shapes

:+@*
	container 
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+@
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+@*
T0
b
training/Adam/zeros_9Const*
_output_shapes
:@*
valueB@*    *
dtype0
�
training/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@*
use_locking(
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:@
w
&training/Adam/zeros_10/shape_as_tensorConst*
valueB"@       *
dtype0*
_output_shapes
:
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*
_output_shapes

:@ 
�
training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*
_output_shapes

:@ *
	container *
shape
:@ 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:@ 
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:@ 
c
training/Adam/zeros_11Const*
valueB *    *
dtype0*
_output_shapes
: 
�
training/Adam/Variable_11
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: 
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
_output_shapes
: *
T0*,
_class"
 loc:@training/Adam/Variable_11
w
&training/Adam/zeros_12/shape_as_tensorConst*
valueB"        *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
T0*

index_type0*
_output_shapes

:  
�
training/Adam/Variable_12
VariableV2*
shared_name *
dtype0*
_output_shapes

:  *
	container *
shape
:  
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
_output_shapes

:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:  
c
training/Adam/zeros_13Const*
valueB *    *
dtype0*
_output_shapes
: 
�
training/Adam/Variable_13
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
: 
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
: *
T0
w
&training/Adam/zeros_14/shape_as_tensorConst*
valueB"    +   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*
_output_shapes

: +
�
training/Adam/Variable_14
VariableV2*
shape
: +*
shared_name *
dtype0*
_output_shapes

: +*
	container 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

: +
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
_output_shapes

: +*
T0*,
_class"
 loc:@training/Adam/Variable_14
c
training/Adam/zeros_15Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_15
VariableV2*
_output_shapes
:+*
	container *
shape:+*
shared_name *
dtype0
�
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+
�
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
_output_shapes
:+*
T0*,
_class"
 loc:@training/Adam/Variable_15
p
&training/Adam/zeros_16/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_16
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
�
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:
p
&training/Adam/zeros_17/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:
p
&training/Adam/zeros_18/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_18
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes
:
p
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_19
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(
�
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
:
p
&training/Adam/zeros_20/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_20
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_21/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_21
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_21
p
&training/Adam/zeros_22/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_22
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
:
p
&training/Adam/zeros_23/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_23/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_23
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes

:+@
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+@
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+@
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:+@
Z
training/Adam/sub_3/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
_output_shapes
: *
T0
}
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+@
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:+@
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

:+@*
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+@
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
_output_shapes

:+@*
T0
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
_output_shapes

:+@*
T0
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:+@
Z
training/Adam/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

:+@
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

:+@*
T0
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
_output_shapes

:+@*
T0
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+@
�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+@
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+@
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:@
Z
training/Adam/sub_5/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:@*
T0
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:@
Z
training/Adam/sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:@
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:@
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:@
Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_5Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
:@
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
:@*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:@
Z
training/Adam/add_6/yConst*
_output_shapes
: *
valueB
 *���3*
dtype0
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:@
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:@*
T0
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
_output_shapes
:@*
T0
�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:@
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:@ 
Z
training/Adam/sub_8/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@ 
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:@ *
T0
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:@ 
Z
training/Adam/sub_9/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:@ 
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
_output_shapes

:@ *
T0
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:@ 
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:@ 
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*
_output_shapes

:@ 
�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes

:@ 
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:@ 
Z
training/Adam/add_9/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes

:@ *
T0
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
_output_shapes

:@ *
T0
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
_output_shapes

:@ *
T0
�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:@ *
use_locking(
�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:@ 
�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:@ 
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
_output_shapes
: *
T0
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
: 
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
: 
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
: 
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
_output_shapes
: *
valueB
 *  �*
dtype0
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
: 
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
: *
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
: *
T0
[
training/Adam/add_12/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
: 
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
: 
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
_output_shapes
: *
T0
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: 
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: *
use_locking(
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense_2/bias
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:  
[
training/Adam/sub_14/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
_output_shapes

:  *
T0
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:  
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
_output_shapes

:  *
T0
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
_output_shapes

:  *
T0
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:  
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes

:  
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
_output_shapes

:  *
T0
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:  
[
training/Adam/add_15/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
_output_shapes

:  *
T0
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:  
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
_output_shapes

:  *
T0
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:  *
use_locking(*
T0
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
validate_shape(*
_output_shapes

:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:  
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
: 
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
: *
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
: 
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
: 
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
: 
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
: 
[
training/Adam/add_18/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes
: *
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
: 
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
: 
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
: 
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
: 
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

: +
[
training/Adam/sub_20/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

: +
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

: +
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

: +
[
training/Adam/sub_21/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

: +
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

: +
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

: +
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

: +
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*
_output_shapes

: +
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

: +
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes

: +*
T0
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes

: +
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

: +
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes

: +
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

: +*
use_locking(
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

: +
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

: +*
use_locking(*
T0
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:+
[
training/Adam/sub_23/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:+
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:+
[
training/Adam/sub_24/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:+
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
_output_shapes
:+*
T0
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:+
[
training/Adam/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_17Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0*
_output_shapes
:+
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
T0*
_output_shapes
:+
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:+
[
training/Adam/add_24/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:+
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes
:+*
T0
l
training/Adam/sub_25Subdense_4/bias/readtraining/Adam/truediv_8*
_output_shapes
:+*
T0
�
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7
�
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+
�
training/Adam/Assign_23Assigndense_4/biastraining/Adam/sub_25*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+
�
training/group_depsNoOp	^loss/add^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9
0

group_depsNoOp	^loss/add^metrics/acc/Mean
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_1/bias
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_output_shapes
: *
_class
loc:@Adam/beta_2*
dtype0
�
IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3
�
IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13*
dtype0
�
IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17*
dtype0
�
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_23*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_23
�
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"F@����     ,Z^	g�{�B=�AJ��
��
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
/
Sign
x"T
y"T"
Ttype:

2	
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.13.12
b'unknown'��
j
input_1Placeholder*
dtype0*'
_output_shapes
:���������+*
shape:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+   @   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�{r�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�{r>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2끞*
_output_shapes

:+@*

seed*
T0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:+@
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+@
�
dense_1/kernel
VariableV2*
shape
:+@*
shared_name *
dtype0*
	container *
_output_shapes

:+@
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
_output_shapes

:+@*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+@
Z
dense_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_1/bias
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:@
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
transpose_a( *'
_output_shapes
:���������@*
transpose_b( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
W
dense_1/TanhTanhdense_1/BiasAdd*'
_output_shapes
:���������@*
T0
g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������@
g
"dense_1/activity_regularizer/mul/xConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/mulMul"dense_1/activity_regularizer/mul/x dense_1/activity_regularizer/Abs*
T0*'
_output_shapes
:���������@
s
"dense_1/activity_regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
g
"dense_1/activity_regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/addAdd"dense_1/activity_regularizer/add/x dense_1/activity_regularizer/Sum*
T0*
_output_shapes
: 
m
dense_2/random_uniform/shapeConst*
valueB"@       *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *  ��*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *  �>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*

seed*
T0*
dtype0*
seed2�ߜ*
_output_shapes

:@ 
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:@ 
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:@ 
�
dense_2/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:@ *
shape
:@ *
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(*
_output_shapes

:@ *
use_locking(*
T0*!
_class
loc:@dense_2/kernel
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:@ 
Z
dense_2/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
x
dense_2/bias
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
: 
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:��������� *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:��������� 
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:��������� 
m
dense_3/random_uniform/shapeConst*
valueB"        *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *qĜ>
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
T0*
dtype0*
seed2��*
_output_shapes

:  *

seed
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:  
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:  
�
dense_3/kernel
VariableV2*
shape
:  *
shared_name *
dtype0*
	container *
_output_shapes

:  
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:  *
use_locking(
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:  
Z
dense_3/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
x
dense_3/bias
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
: 
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
T0*
transpose_a( *'
_output_shapes
:��������� *
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:��������� *
T0
W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:��������� 
m
dense_4/random_uniform/shapeConst*
valueB"    +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�А�
_
dense_4/random_uniform/maxConst*
valueB
 *�А>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
seed2�Ֆ*
_output_shapes

: +*

seed*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

: +*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

: +
�
dense_4/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

: +*
shape
: +
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

: +
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

: +
Z
dense_4/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
x
dense_4/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:+*
shape:+
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:+
�
dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*
transpose_a( *'
_output_shapes
:���������+*
transpose_b( *
T0
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������+
W
dense_4/ReluReludense_4/BiasAdd*
T0*'
_output_shapes
:���������+
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
dtype0	*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
_output_shapes
: *
T0
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
Adam/beta_1
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
_class
loc:@Adam/beta_1*
_output_shapes
: *
T0
^
Adam/beta_2/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
�
dense_4_targetPlaceholder*0
_output_shapes
:������������������*%
shape:������������������*
dtype0
q
dense_4_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
l
loss/dense_4_loss/subSubdense_4/Reludense_4_target*
T0*'
_output_shapes
:���������+
k
loss/dense_4_loss/SquareSquareloss/dense_4_loss/sub*'
_output_shapes
:���������+*
T0
s
(loss/dense_4_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
m
*loss/dense_4_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Mean*loss/dense_4_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
|
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*#
_output_shapes
:���������*
T0
a
loss/dense_4_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*#
_output_shapes
:���������*
T0
�
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
a
loss/dense_4_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_2*
T0*#
_output_shapes
:���������
c
loss/dense_4_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_3Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
loss/mulMul
loss/mul/xloss/dense_4_loss/Mean_3*
T0*
_output_shapes
: 
\
loss/addAddloss/mul dense_1/activity_regularizer/add*
_output_shapes
: *
T0
g
metrics/acc/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMaxArgMaxdense_4_targetmetrics/acc/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:���������*
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*
Truncate( *

DstT0*#
_output_shapes
:���������*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/add*
valueB *
dtype0*
_output_shapes
: 
�
!training/Adam/gradients/grad_ys_0Const*
_class
loc:@loss/add*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/add*

index_type0*
_output_shapes
: 
�
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_3*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Tshape0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1Shapeloss/dense_4_loss/truediv*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2Const*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB *
dtype0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ConstConst*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
valueB"      *
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
_output_shapes

:*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
Tshape0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
out_type0*
_output_shapes
:
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*'
_output_shapes
:���������@*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
_output_shapes
:*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
out_type0
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*,
_class"
 loc:@loss/dense_4_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:���������:���������
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������*
T0
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *3
_class)
'%loc:@dense_1/activity_regularizer/mul*
valueB 
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1Shape dense_1/activity_regularizer/Abs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
out_type0*
_output_shapes
:
�
Straining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeEtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*2
_output_shapes 
:���������:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������@*
T0
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������@*
T0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Utraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*'
_output_shapes
:���������@
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*2
_output_shapes 
:���������:���������
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*#
_output_shapes
:���������*
T0
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*#
_output_shapes
:���������
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������@
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������@
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/addAdd*loss/dense_4_loss/Mean_1/reduction_indices:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/startConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*

Tidx0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/valueConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*

index_type0
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
N*
_output_shapes
:
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/MaximumMaximumCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordivFloorDiv;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Tshape0*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Reshape>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2Shapeloss/dense_4_loss/Mean*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: 
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
T0
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/CastCast@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Truncate( *

DstT0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*#
_output_shapes
:���������*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeloss/dense_4_loss/Square*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
valueB *
dtype0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*

Tidx0
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*)
_class
loc:@loss/dense_4_loss/Mean*

index_type0*
_output_shapes
: *
T0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
N*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*)
_class
loc:@loss/dense_4_loss/Mean*
Tshape0*0
_output_shapes
:������������������*
T0
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2Shapeloss/dense_4_loss/Square*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:*
T0
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_4_loss/Mean*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/ConstConst<^training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Square*
valueB
 *   @*
dtype0
�
9training/Adam/gradients/loss/dense_4_loss/Square_grad/MulMulloss/dense_4_loss/sub;training/Adam/gradients/loss/dense_4_loss/Square_grad/Const*'
_output_shapes
:���������+*
T0*+
_class!
loc:@loss/dense_4_loss/Square
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Mul;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv9training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/ShapeShapedense_4/Relu*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0*
_output_shapes
:
�
Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*2
_output_shapes 
:���������:���������
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/SumSum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape*'
_output_shapes
:���������+*
T0*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*0
_output_shapes
:������������������
�
2training/Adam/gradients/dense_4/Relu_grad/ReluGradReluGrad:training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshapedense_4/Relu*
T0*
_class
loc:@dense_4/Relu*'
_output_shapes
:���������+
�
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC*
_output_shapes
:+
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a( *'
_output_shapes
:��������� 
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a(*
_output_shapes

: +*
transpose_b( 
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*
T0*
_class
loc:@dense_3/Tanh*'
_output_shapes
:��������� 
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
: *
T0*"
_class
loc:@dense_3/BiasAdd
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( *'
_output_shapes
:��������� 
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(*
_output_shapes

:  *
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:��������� 
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
: 
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*!
_class
loc:@dense_2/MatMul*
transpose_a( *'
_output_shapes
:���������@*
transpose_b(*
T0
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(*
_output_shapes

:@ 
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������@*
T0
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������@
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:@*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
transpose_a( *'
_output_shapes
:���������+*
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*!
_class
loc:@dense_1/MatMul*
transpose_a(*
_output_shapes

:+@*
transpose_b( *
T0
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
p
training/Adam/CastCastAdam/iterations/read*

SrcT0	*
Truncate( *

DstT0*
_output_shapes
: 
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
t
#training/Adam/zeros/shape_as_tensorConst*
valueB"+   @   *
dtype0*
_output_shapes
:
^
training/Adam/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zerosFill#training/Adam/zeros/shape_as_tensortraining/Adam/zeros/Const*
_output_shapes

:+@*
T0*

index_type0
�
training/Adam/Variable
VariableV2*
shape
:+@*
shared_name *
dtype0*
	container *
_output_shapes

:+@
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+@
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+@
b
training/Adam/zeros_1Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
training/Adam/Variable_1
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:@
v
%training/Adam/zeros_2/shape_as_tensorConst*
valueB"@       *
dtype0*
_output_shapes
:
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*
_output_shapes

:@ 
�
training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:@ *
shape
:@ 
�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:@ 
�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
_output_shapes

:@ *
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
valueB *    *
dtype0*
_output_shapes
: 
�
training/Adam/Variable_3
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: 
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
: 
v
%training/Adam/zeros_4/shape_as_tensorConst*
valueB"        *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*
_output_shapes

:  
�
training/Adam/Variable_4
VariableV2*
shape
:  *
shared_name *
dtype0*
	container *
_output_shapes

:  
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:  
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:  
b
training/Adam/zeros_5Const*
valueB *    *
dtype0*
_output_shapes
: 
�
training/Adam/Variable_5
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
: 
v
%training/Adam/zeros_6/shape_as_tensorConst*
valueB"    +   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*
_output_shapes

: +
�
training/Adam/Variable_6
VariableV2*
shape
: +*
shared_name *
dtype0*
	container *
_output_shapes

: +
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
validate_shape(*
_output_shapes

: +*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes

: +*
T0*+
_class!
loc:@training/Adam/Variable_6
b
training/Adam/zeros_7Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_7
VariableV2*
shape:+*
shared_name *
dtype0*
	container *
_output_shapes
:+
�
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7
�
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:+
v
%training/Adam/zeros_8/shape_as_tensorConst*
valueB"+   @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*
_output_shapes

:+@
�
training/Adam/Variable_8
VariableV2*
dtype0*
	container *
_output_shapes

:+@*
shape
:+@*
shared_name 
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+@*
use_locking(
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+@
b
training/Adam/zeros_9Const*
valueB@*    *
dtype0*
_output_shapes
:@
�
training/Adam/Variable_9
VariableV2*
dtype0*
	container *
_output_shapes
:@*
shape:@*
shared_name 
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_9
w
&training/Adam/zeros_10/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"@       
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*
_output_shapes

:@ 
�
training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:@ *
shape
:@ 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:@ 
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
_output_shapes

:@ *
T0*,
_class"
 loc:@training/Adam/Variable_10
c
training/Adam/zeros_11Const*
valueB *    *
dtype0*
_output_shapes
: 
�
training/Adam/Variable_11
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: 
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: 
w
&training/Adam/zeros_12/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"        
a
training/Adam/zeros_12/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
_output_shapes

:  *
T0*

index_type0
�
training/Adam/Variable_12
VariableV2*
dtype0*
	container *
_output_shapes

:  *
shape
:  *
shared_name 
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:  *
use_locking(
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:  
c
training/Adam/zeros_13Const*
dtype0*
_output_shapes
: *
valueB *    
�
training/Adam/Variable_13
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
: 
w
&training/Adam/zeros_14/shape_as_tensorConst*
valueB"    +   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*
_output_shapes

: +
�
training/Adam/Variable_14
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

: +*
shape
: +
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

: +
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

: +
c
training/Adam/zeros_15Const*
valueB+*    *
dtype0*
_output_shapes
:+
�
training/Adam/Variable_15
VariableV2*
shape:+*
shared_name *
dtype0*
	container *
_output_shapes
:+
�
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+
�
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:+
p
&training/Adam/zeros_16/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_16
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
�
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_16
p
&training/Adam/zeros_17/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_17/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_17
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(
�
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
_output_shapes
:*
T0
p
&training/Adam/zeros_18/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_18
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes
:*
T0
p
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_19
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_19
p
&training/Adam/zeros_20/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_20/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_20
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_21/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_21
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes
:
p
&training/Adam/zeros_22/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_22/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_22
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
:*
T0
p
&training/Adam/zeros_23/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_23/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_23
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23
�
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes

:+@
Z
training/Adam/sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+@
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes

:+@*
T0
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:+@
Z
training/Adam/sub_3/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
}
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+@
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes

:+@*
T0
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes

:+@
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+@
Z
training/Adam/Const_2Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_3Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
_output_shapes

:+@*
T0
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
_output_shapes

:+@*
T0
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
_output_shapes

:+@*
T0
Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes

:+@
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes

:+@
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
_output_shapes

:+@*
T0
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+@*
use_locking(
�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
_output_shapes

:+@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+@
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
:@*
T0
Z
training/Adam/sub_5/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:@
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:@
Z
training/Adam/sub_6/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:@
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:@
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes
:@*
T0
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
:@
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:@
Z
training/Adam/add_6/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:@
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:@*
T0
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:@
�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:@
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
_output_shapes

:@ *
T0
Z
training/Adam/sub_8/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@ *
T0
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:@ *
T0
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
_output_shapes

:@ *
T0
Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@ *
T0
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:@ 
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:@ 
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:@ 
Z
training/Adam/Const_6Const*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_7Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
T0*
_output_shapes

:@ 
�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
_output_shapes

:@ *
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:@ 
Z
training/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *���3*
dtype0
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes

:@ 
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:@ 
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:@ 
�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:@ 
�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:@ *
use_locking(*
T0
�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:@ *
use_locking(
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes
: *
T0
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
_output_shapes
: *
T0
[
training/Adam/sub_12/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
: *
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes
: *
T0
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
: 
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
_output_shapes
: *
T0
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
: *
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
: *
T0
[
training/Adam/add_12/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
: 
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes
: *
T0
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
_output_shapes
: *
T0
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: 
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
: 
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:  
[
training/Adam/sub_14/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:  
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
_output_shapes

:  *
T0
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:  
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:  
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:  
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:  
[
training/Adam/Const_10Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_11Const*
_output_shapes
: *
valueB
 *  �*
dtype0
�
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*
_output_shapes

:  
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:  
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:  
[
training/Adam/add_15/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
_output_shapes

:  *
T0
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:  
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:  
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:  *
use_locking(*
T0
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
validate_shape(*
_output_shapes

:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
validate_shape(*
_output_shapes

:  *
use_locking(*
T0*!
_class
loc:@dense_3/kernel
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
: 
[
training/Adam/sub_17/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
: 
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
: 
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
: 
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
: 
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes
: *
T0
[
training/Adam/Const_12Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_13Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes
: *
T0
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes
: *
T0
[
training/Adam/add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes
: *
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
: *
T0
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
: 
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
_output_shapes
: *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
: 
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
: 
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

: +
[
training/Adam/sub_20/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

: +
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

: +
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

: +
[
training/Adam/sub_21/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

: +
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes

: +*
T0
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes

: +*
T0
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

: +
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
_output_shapes

: +*
T0
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

: +
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes

: +*
T0
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes

: +
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

: +
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

: +*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

: +
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
validate_shape(*
_output_shapes

: +*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
_output_shapes

: +*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
_output_shapes
:+*
T0
[
training/Adam/sub_23/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:+*
T0
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:+
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
_output_shapes
:+*
T0
[
training/Adam/sub_24/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:+
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:+
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:+*
T0
[
training/Adam/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_17Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0*
_output_shapes
:+
�
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
_output_shapes
:+*
T0
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:+
[
training/Adam/add_24/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:+
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:+
l
training/Adam/sub_25Subdense_4/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes
:+
�
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:+
�
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
_output_shapes
:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(
�
training/Adam/Assign_23Assigndense_4/biastraining/Adam/sub_25*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0
�
training/group_depsNoOp	^loss/add^metrics/acc/Mean^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9
0

group_depsNoOp	^loss/add^metrics/acc/Mean
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_5*
dtype0
�
IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7
�
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*,
_class"
 loc:@training/Adam/Variable_10*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_22*,
_class"
 loc:@training/Adam/Variable_22*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 
�
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign""� 
trainable_variables� � 
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
\
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02dense_4/random_uniform:08
M
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02dense_4/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08"� 
	variables� � 
\
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:08
M
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:08
\
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:08
M
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:08
\
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:08
M
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:08
\
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02dense_4/random_uniform:08
M
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02dense_4/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08ܝ�       ���	)!��B=�A*

val_loss�O,@��        �	�"��B=�A*

val_acc��>#r��       �K"	�#��B=�A*

loss�Em@<5       ���	-$��B=�A*


acc~�O>	�j       ��2	����B=�A*

val_lossMz�?�)ot       `/�#	���B=�A*

val_acc])�>�D�       ��-	t���B=�A*

loss�@+�Q�       ��(	̹��B=�A*


acc�X�>��#       ��2	�G��B=�A*

val_lossc0|?�v��       `/�#	�H��B=�A*

val_acc�y�>�l(�       ��-	NI��B=�A*

lossFF�?�D�C       ��(	�I��B=�A*


accG�>e�}       ��2	ӢȾB=�A*

val_loss�i[?89�p       `/�#	U�ȾB=�A*

val_accJ��>]J�       ��-	�ȾB=�A*

lossR�q?�!       ��(	��ȾB=�A*


acc���>d]�x       ��2	��ԾB=�A*

val_lossVI?�U6�       `/�#	�ԾB=�A*

val_acc���>h�=R       ��-	ÂԾB=�A*

lossf<X?��       ��(	4�ԾB=�A*


accS��>! &B       ��2	+�B=�A*

val_lossc�=?EN�       `/�#	
,�B=�A*

val_acc;��>�c�       ��-	s,�B=�A*

loss@HH?`�	       ��(	�,�B=�A*


acc��>B��       ��2	�O��B=�A*

val_lossl�4?��m       `/�#	�P��B=�A*

val_accAr�>��^       ��-	DQ��B=�A*

lossY
=?mt�       ��(	�Q��B=�A*


acc���>��6       ��2	���B=�A*

val_loss�x.?��+       `/�#	���B=�A*

val_acc���>�8�       ��-	y���B=�A*

loss\v4?��	�       ��(	ѕ��B=�A*


acc�B�>�H��       ��2	h�B=�A*

val_loss��(?�v��       `/�#	|�B=�A*

val_acc;��>��y       ��-	��B=�A*

loss�?.?"�       ��(	J	�B=�A*


acc�w�>��M       ��2	K�B=�A	*

val_lossM>$?#�2       `/�#	e�B=�A	*

val_acc+�>�P�       ��-	��B=�A	*

loss	 )?Y�/
       ��(	)�B=�A	*


acc��>�%��       ��2	��B=�A
*

val_loss[� ?�<�@       `/�#	���B=�A
*

val_acc��>&�L�       ��-	 ��B=�A
*

loss�%?���       ��(	X��B=�A
*


acc���>�)�       ��2	��*�B=�A*

val_loss_?y-b       `/�#	ۆ*�B=�A*

val_acc$��>2�       ��-	H�*�B=�A*

loss�B!?�r�       ��(	��*�B=�A*


accL��>p�j       ��2	ծ7�B=�A*

val_loss�?P���       `/�#	%�7�B=�A*

val_acc�e�>_��F       ��-	��7�B=�A*

loss �?����       ��(	)�7�B=�A*


accn��>�|�       ��2	�aD�B=�A*

val_loss`v?A�O�       `/�#	#cD�B=�A*

val_accw��>�T"       ��-	�cD�B=�A*

loss�?����       ��(	=dD�B=�A*


acc/�>�-       ��2	D-P�B=�A*

val_loss>d?`o8       `/�#	�.P�B=�A*

val_acc�"?:|G�       ��-	0P�B=�A*

loss�J?rD�,       ��(	�0P�B=�A*


acc�?*r       ��2	�\�B=�A*

val_loss�=?*��$       `/�#	ʩ\�B=�A*

val_acc	?��υ       ��-	q�\�B=�A*

loss�y?c8       ��(	֪\�B=�A*


acc��?;C�^       ��2	�;i�B=�A*

val_loss�?����       `/�#	=i�B=�A*

val_acc}7�>�f�       ��-	�=i�B=�A*

lossl�?ƄZs       ��(	2>i�B=�A*


acc`�?1�       ��2	T}�B=�A*

val_loss��?��_�       `/�#	:}�B=�A*

val_acc�R?x7Yu       ��-	�}�B=�A*

loss2C?��o�       ��(	-}�B=�A*


accΪ	?1�ew       ��2	�9��B=�A*

val_loss�?p�       `/�#	�;��B=�A*

val_accDW
?�l?v       ��-	�<��B=�A*

loss�L?'�,       ��(	K=��B=�A*


accu�?Aӄ;       ��2	g��B=�A*

val_lossQ8?^N5�       `/�#	Qh��B=�A*

val_acc��?nXB       ��-	�i��B=�A*

lossy�?IR	�       ��(	�j��B=�A*


acc`�?B���       ��2	����B=�A*

val_lossKh?+,%       `/�#		���B=�A*

val_acc�?���m       ��-	����B=�A*

loss??^x^       ��(	����B=�A*


acc�?���       ��2	y��B=�A*

val_loss��?둏�       `/�#	C��B=�A*

val_acc�?���       ��-	���B=�A*

loss��?e
Rw       ��(	`��B=�A*


accS?�(�$       ��2	X�ѿB=�A*

val_lossҵ?�o�Y       `/�#	&�ѿB=�A*

val_acc"�	?-f�       ��-	��ѿB=�A*

loss�?F��&       ��(	�ѿB=�A*


acc��?���       ��2	�ݿB=�A*

val_loss��?6��       `/�#	��ݿB=�A*

val_accx�?E��       ��-	ޒݿB=�A*

loss��?�&        ��(	��ݿB=�A*


acc��?Bܪd       ��2	�H�B=�A*

val_loss1?I6�       `/�#	AK�B=�A*

val_accUB?t��@       ��-	L�B=�A*

loss
�?��'       ��(	M�B=�A*


accai?�E�       ��2	��B=�A*

val_loss��?��]       `/�#	���B=�A*

val_acc\�? l!       ��-	c��B=�A*

losso$?m��       ��(	���B=�A*


acc9?�"��       ��2	�a�B=�A*

val_loss.�?RX��       `/�#	�b�B=�A*

val_acc��?$X�O       ��-	c�B=�A*

loss@p?�Z�       ��(	oc�B=�A*


acc��?e��2       ��2	�/ �B=�A*

val_loss}�?����       `/�#	�0 �B=�A*

val_accf-?�S       ��-	a1 �B=�A*

loss�?�Y�U       ��(	�1 �B=�A*


acc�?���       ��2	�-�B=�A*

val_loss�=
?�D�       `/�#	[-�B=�A*

val_acc��?�g�4       ��-	�-�B=�A*

lossC�?h���       ��(	_-�B=�A*


acc�U?.(�#       ��2	�+<�B=�A*

val_loss�	?\a#h       `/�#	�-<�B=�A*

val_acct�?H���       ��-	/<�B=�A*

loss��
?{�D~       ��(	�/<�B=�A*


acc��?QXI3       ��2	k�J�B=�A*

val_loss�0	?�f_�       `/�#	k�J�B=�A*

val_acc\�?�R$+       ��-	��J�B=�A*

loss��	?��i�       ��(	,�J�B=�A*


acc<� ?g�-        ��2	��T�B=�A*

val_loss#k	?#'�       `/�#	��T�B=�A*

val_accUB?�+�F       ��-	R�T�B=�A*

loss��	?���q       ��(	��T�B=�A*


acc�?)�%       ��2	�c�B=�A *

val_loss�?��       `/�#	�c�B=�A *

val_acc�?8��       ��-	��c�B=�A *

lossTO	?\�<       ��(	
�c�B=�A *


acc7�?t���       ��2	�l�B=�A!*

val_losso�?_��       `/�#	� l�B=�A!*

val_accw?C�N�       ��-	W!l�B=�A!*

loss.�?SqQ�       ��(	�!l�B=�A!*


acc�(?��I       ��2	��{�B=�A"*

val_loss�+?��9%       `/�#	2�{�B=�A"*

val_acc�?X��F       ��-	��{�B=�A"*

lossa|?�`�       ��(	�{�B=�A"*


acc$"?v��       ��2	>��B=�A#*

val_loss�r?�:3�       `/�#	1?��B=�A#*

val_accz�!?�Rx       ��-	�?��B=�A#*

lossq~?(ib�       ��(	�?��B=�A#*


acc "?�3*U       ��2	�ј�B=�A$*

val_loss�?�թ�       `/�#	�Ҙ�B=�A$*

val_acc?�K:�       ��-	5Ә�B=�A$*

loss��?�z�       ��(	�Ә�B=�A$*


acc��?A�       ��2	��B=�A%*

val_loss�o?� e�       `/�#	���B=�A%*

val_acc!+?��#       ��-	^��B=�A%*

loss��?����       ��(	���B=�A%*


accZ0"?uP{       ��2	>>��B=�A&*

val_loss�u?�D3       `/�#	g?��B=�A&*

val_acc�?s�       ��-	�?��B=�A&*

loss6�?Xv�       ��(	1@��B=�A&*


accN("?I��G       ��2	�I��B=�A'*

val_loss�q?��_       `/�#	K��B=�A'*

val_acc?�?��))       ��-	�K��B=�A'*

loss�X?�ɯ       ��(	L��B=�A'*


acc "?�I�!       ��2	Nc��B=�A(*

val_loss��?���K       `/�#	Ad��B=�A(*

val_accp�?3+Ҧ       ��-	�d��B=�A(*

loss��?���o       ��(	e��B=�A(*


accl?D{x       ��2	X��B=�A)*

val_lossU?=�,�       `/�#	T!��B=�A)*

val_acc'$?�oD       ��-	�!��B=�A)*

loss��?ur�S       ��(	S"��B=�A)*


acc��?���4       ��2	����B=�A**

val_losskQ?"2�       `/�#	<���B=�A**

val_acc�h ?9��       ��-	���B=�A**

loss�k?PM{4       ��(	����B=�A**


acc!�%?�E.�       ��2	|��B=�A+*

val_loss�?lI�       `/�#	}��B=�A+*

val_acc�J!?�Id       ��-	�}��B=�A+*

loss=�?� �2       ��(	�}��B=�A+*


acc{;$?��ա       ��2	qX
�B=�A,*

val_loss��?�A_C       `/�#	�\
�B=�A,*

val_acc��"?7���       ��-	�^
�B=�A,*

loss4?N^�.       ��(	�`
�B=�A,*


acc{;$?fq�       ��2	���B=�A-*

val_loss\ ?���       `/�#	���B=�A-*

val_accY?dk|�       ��-	,��B=�A-*

loss�4?o+N       ��(	���B=�A-*


acc�#?�_k       ��2	��&�B=�A.*

val_lossv�?;�w       `/�#	?�&�B=�A.*

val_accQy?j���       ��-	��&�B=�A.*

lossސ?�X��       ��(	K�&�B=�A.*


acck&&?�|��       ��2	w�5�B=�A/*

val_lossĭ?���       `/�#	��5�B=�A/*

val_accBV&?mD�\       ��-	�5�B=�A/*

loss:�?�
��       ��(	Y�5�B=�A/*


acc��?c	
r       ��2	�G�B=�A0*

val_loss��?l%�d       `/�#	�G�B=�A0*

val_accI�$?�}       ��-	�	G�B=�A0*

lossz�?���       ��(	[
G�B=�A0*


accM�*?U]��       ��2	{�W�B=�A1*

val_lossvi?� �8       `/�#	кW�B=�A1*

val_acc�"?m��       ��-	��W�B=�A1*

lossu>?w�       ��(	M�W�B=�A1*


acc�$*?oK0�       ��2	2Wg�B=�A2*

val_loss�D?���       `/�#	�Xg�B=�A2*

val_acc1k!?����       ��-	\Yg�B=�A2*

lossrd?Lԥ�       ��(	�Yg�B=�A2*


acc�.?j7��       ��2	cFy�B=�A3*

val_loss ?��       `/�#	�Ky�B=�A3*

val_acc]�1?�Dk       ��-	�My�B=�A3*

loss�??Az�       ��(	�Oy�B=�A3*


acc=�,?�PG�       ��2	\���B=�A4*

val_lossY�?M��6       `/�#	����B=�A4*

val_acc�,"?Tٌ�       ��-	����B=�A4*

loss[0?��,�       ��(	`���B=�A4*


acc�D*?|5\�       ��2	b0��B=�A5*

val_lossm�?uY       `/�#	]1��B=�A5*

val_acc�&?A�~}       ��-	�1��B=�A5*

loss`�?M��       ��(	U2��B=�A5*


accr�*? �(�       ��2	���B=�A6*

val_loss~7? ��       `/�#	 ���B=�A6*

val_acc��?	^��       ��-	u���B=�A6*

losss?�g7       ��(	����B=�A6*


acc�i-?ߖJ�       ��2	���B=�A7*

val_loss_`?͗2       `/�#	���B=�A7*

val_acc�)?��+       ��-	d��B=�A7*

loss�O?K�}<       ��(	���B=�A7*


acc��+?s���       ��2	C��B=�A8*

val_lossW�?�,G�       `/�#	C��B=�A8*

val_accBV&?��V�       ��-	� ��B=�A8*

loss"?�|t�       ��(	[$��B=�A8*


acc/$/?��P       ��2	����B=�A9*

val_lossA�?]ik�       `/�#	~���B=�A9*

val_acc4�)?6p.�       ��-	#���B=�A9*

loss_?C�       ��(	����B=�A9*


accܔ/?��?r       ��2	����B=�A:*

val_loss��?=�W�       `/�#	����B=�A:*

val_acc�)?CO/�       ��-	���B=�A:*

lossQ�?±\�       ��(	|���B=�A:*


acc�-0?D���       ��2	Ԟ��B=�A;*

val_loss٧?��n       `/�#	����B=�A;*

val_accE�.?;��8       ��-	k���B=�A;*

loss��?r��       ��(	ʠ��B=�A;*


acc�*.?���       ��2	}<��B=�A<*

val_loss�?I�.�       `/�#	�=��B=�A<*

val_accL-?^9�       ��-	C>��B=�A<*

loss\l?��       ��(	�>��B=�A<*


acc I2?���~       ��2	�D�B=�A=*

val_loss��?S�$_       `/�#	�E�B=�A=*

val_accq�5?36��       ��-	=F�B=�A=*

loss�?<��       ��(	�F�B=�A=*


acc�1?�       ��2	9��B=�A>*

val_loss{$?�m�       `/�#	=��B=�A>*

val_acc�'.?���
       ��-	���B=�A>*

loss	C??.       ��(	��B=�A>*


acc�W1?n~��       ��2	�0#�B=�A?*

val_loss�j?9�S�       `/�#	f2#�B=�A?*

val_acc�+?&��.       ��-	�2#�B=�A?*

loss*F?C,��       ��(	H3#�B=�A?*


acc��2?`��       ��2	��7�B=�A@*

val_lossѿ?��w       `/�#	��7�B=�A@*

val_acc>�0?�_w       ��-	P�7�B=�A@*

loss��?��a       ��(	��7�B=�A@*


acc��2?
/E       ��2	�WC�B=�AA*

val_lossĳ?;��H       `/�#	�YC�B=�AA*

val_acc��$?���}       ��-	�ZC�B=�AA*

loss�F?b��       ��(	[C�B=�AA*


acc�+?�*��       ��2	��R�B=�AB*

val_loss�?bG��       `/�#	��R�B=�AB*

val_accY�2?����       ��-	�R�B=�AB*

loss"�?��F�       ��(	d�R�B=�AB*


acc�X7?�<O       ��2	��`�B=�AC*

val_lossh�?K�J�       `/�#	p�`�B=�AC*

val_acc��?Fa?       ��-	ɔ`�B=�AC*

lossj�?�`�       ��(	c�`�B=�AC*


acc��5?���        ��2	�dn�B=�AD*

val_loss�&?U-2�       `/�#	�en�B=�AD*

val_acc��)?n��       ��-	fn�B=�AD*

loss��?�8�b       ��(	^fn�B=�AD*


acc1?���       ��2	E�B=�AE*

val_loss�=?��{~       `/�#	g�B=�AE*

val_acc�N1?�a�       ��-	��B=�AE*

loss�O?�rj]       ��(	/�B=�AE*


acc��4?�4�       ��2	����B=�AF*

val_loss��?�X%C       `/�#	���B=�AF*

val_acc��.?�t�       ��-	����B=�AF*

lossy�?!��       ��(	4���B=�AF*


acc�0?
�       ��2	���B=�AG*

val_lossd�?��Q       `/�#	��B=�AG*

val_acc��4?u0D�       ��-	���B=�AG*

loss}�?#%߭       ��(	���B=�AG*


acc)O6?��>       ��2	b��B=�AH*

val_loss� ?��א       `/�#	f���B=�AH*

val_accC:?A��=       ��-	w���B=�AH*

loss�g?h-��       ��(	f���B=�AH*


acc�e5?�?�       ��2	�S��B=�AI*

val_loss�w?�`       `/�#	�T��B=�AI*

val_acc��2?����       ��-	"U��B=�AI*

loss��?)s"�       ��(	�U��B=�AI*


acc�-0?�表       ��2	+��B=�AJ*

val_loss/?�?�a       `/�#	]��B=�AJ*

val_acc�2?��L       ��-	���B=�AJ*

loss�H?��z       ��(	#��B=�AJ*


acc��4?�`�       ��2	$*��B=�AK*

val_loss� ?JL       `/�#	�,��B=�AK*

val_acc�2?7=��       ��-	�-��B=�AK*

loss�� ?��9       ��(	8.��B=�AK*


accr9?T��       ��2	����B=�AL*

val_lossջ?Z�       `/�#	e���B=�AL*

val_acc"H.?U*:$       ��-	����B=�AL*

loss�� ?qB{=       ��(	L���B=�AL*


acc��6? ��*       ��2	}���B=�AM*

val_loss~&?ʿ��       `/�#	���B=�AM*

val_acc�^8?ٝ��       ��-	���B=�AM*

loss�� ?Ҡ��       ��(	���B=�AM*


accɷ6?�F7�       ��2	��B=�AN*

val_lossf ?n��/       `/�#	 ��B=�AN*

val_acc�2?���v       ��-	q ��B=�AN*

loss�� ?� )       ��(	� ��B=�AN*


acc�Z3?���       ��2	�	�B=�AO*

val_lossP� ?�Dl       `/�#	�	�B=�AO*

val_accj�7?P��       ��-	_
	�B=�AO*

loss�� ?��       ��(	�	�B=�AO*


acc�+9?Ҿ`�       ��2	���B=�AP*

val_lossT? J��       `/�#	���B=�AP*

val_accȑ2?.F�       ��-	u��B=�AP*

loss�? ?����       ��(	���B=�AP*


acc:E:?��#J       ��2	�4%�B=�AQ*

val_loss� ?��b�       `/�#	�7%�B=�AQ*

val_acc�3?X�       ��-	m9%�B=�AQ*

loss�� ?���       ��(	�:%�B=�AQ*


accV�2?_�       ��2	��2�B=�AR*

val_lossY� ?W��q       `/�#	��2�B=�AR*

val_acc��1?��+�       ��-	��2�B=�AR*

loss�3 ?���s       ��(	��2�B=�AR*


accFM:?��F�       ��2	f�>�B=�AS*

val_loss�� ?�KP       `/�#	~�>�B=�AS*

val_accu5?�@�       ��-	"�>�B=�AS*

lossԊ ?�yw�       ��(	��>�B=�AS*


acc28?�'"k       ��2	��L�B=�AT*

val_loss� ?h|�=       `/�#	�L�B=�AT*

val_acc�@9?�;8�       ��-	ƅL�B=�AT*

lossD ?;�X       ��(	"�L�B=�AT*


acc�07?�̼0       ��2	�Z�B=�AU*

val_loss�?!mn�       `/�#	fZ�B=�AU*

val_acc7�(?�G�=       ��-	�Z�B=�AU*

loss6 ?K�؁       ��(	UZ�B=�AU*


accA�8?a�bn       ��2	�k�B=�AV*

val_loss�?6��       `/�#	�k�B=�AV*

val_acch�%?J�f       ��-	�k�B=�AV*

loss�Y ?��/V       ��(	 k�B=�AV*


acc��4?�)T�       ��2	 �u�B=�AW*

val_lossa�?iG�       `/�#	��u�B=�AW*

val_acc�c,?Ɲ^!       ��-	u�u�B=�AW*

loss�I ?�9�v       ��(	��u�B=�AW*


accz55?�t�       ��2	o���B=�AX*

val_loss�< ?�=��       `/�#	���B=�AX*

val_acc�0?��%       ��-	����B=�AX*

loss���>���       ��(	����B=�AX*


acc�5?�hD�       ��2	0���B=�AY*

val_loss�? ?��Dk       `/�#	8���B=�AY*

val_acc"�7?�|}P       ��-	����B=�AY*

loss� ?��yS       ��(	+���B=�AY*


acc��8?!I��       ��2	iZ��B=�AZ*

val_loss�x�> r[+       `/�#	d[��B=�AZ*

val_acc�B?�/$E       ��-	�[��B=�AZ*

loss���>n}$�       ��(	%\��B=�AZ*


acc}�;?�sI       ��2	���B=�A[*

val_lossA��>�v�N       `/�#	q���B=�A[*

val_accY'<?��d       ��-	���B=�A[*

loss���>��       ��(	u���B=�A[*


accR�<?b�,       ��2	U��B=�A\*

val_loss���>�s��       `/�#	V��B=�A\*

val_acc)�5?�֛�       ��-	�V��B=�A\*

loss���>����       ��(	�V��B=�A\*


acc�V;?�/j�       ��2	.@��B=�A]*

val_loss���>����       `/�#	%A��B=�A]*

val_acc��=?��       ��-	�A��B=�A]*

loss�9�>����       ��(	B��B=�A]*


acc/�=?HYA�       ��2	I���B=�A^*

val_loss�u�>��i�       `/�#	8���B=�A^*

val_acc�E-?�U       ��-	����B=�A^*

loss� ??�       ��(	���B=�A^*


acc�_1?��g       ��2	<���B=�A_*

val_loss���>Wu�       `/�#	����B=�A_*

val_accH<7?�Q��       ��-	{���B=�A_*

loss^�>�Q�*       ��(	����B=�A_*


accWC>?+�Υ       ��2	���B=�A`*

val_losszD�>P6G�       `/�#	����B=�A`*

val_acctl>?��V	       ��-	����B=�A`*

loss���>ZF�       ��(	����B=�A`*


accғ>?dph�       ��2	T��B=�Aa*

val_loss���>���~       `/�#	3��B=�Aa*

val_acc�3?1�5�       ��-	���B=�Aa*

loss?�>�bV�       ��(	���B=�Aa*


acc�>;?l��H       ��2	@��B=�Ab*

val_loss���>b��       `/�#	U��B=�Ab*

val_accL-?aV9       ��-	���B=�Ab*

loss`�>$b�d       ��(	��B=�Ab*


accT�=?`���       ��2	ob"�B=�Ac*

val_lossc��>��       `/�#	�c"�B=�Ac*

val_acc�;?���{       ��-	�d"�B=�Ac*

loss^��>j���       ��(	be"�B=�Ac*


acc��=?3ܵ�       ��2	$�3�B=�Ad*

val_loss�>���       `/�#	1�3�B=�Ad*

val_accQ�=?�"�w       ��-	��3�B=�Ad*

loss"f�>'�7       ��(	�3�B=�Ad*


acc\�9?���       ��2	�C�B=�Ae*

val_loss���>[���       `/�#	�C�B=�Ae*

val_acc{�<?~�A�       ��-	��C�B=�Ae*

loss.U�>&�ߖ       ��(	K�C�B=�Ae*


accғ>?<2QF       ��2	غO�B=�Af*

val_loss�0 ?ꦑI       `/�#	˻O�B=�Af*

val_acc�^8?-��       ��-	@�O�B=�Af*

loss �>�	��       ��(	��O�B=�Af*


accH�=?[4`�       ��2	��_�B=�Ag*

val_loss���>)�!       `/�#	��_�B=�Ag*

val_acc�<?,H       ��-	J�_�B=�Ag*

loss�~�>k_/       ��(	��_�B=�Ag*


acc�p<?m��       ��2	Ɔn�B=�Ah*

val_lossD ?�v�       `/�#	ׇn�B=�Ah*

val_acc �-?o�        ��-	H�n�B=�Ah*

loss4��>��       ��(	��n�B=�Ah*


acc�=?+e�       ��2	՘|�B=�Ai*

val_lossQ��>Bd       `/�#	M�|�B=�Ai*

val_acc�N1?/ �       ��-	�|�B=�Ai*

lossJ��>'.V�       ��(	R�|�B=�Ai*


accI5?{��
       ��2	�e��B=�Aj*

val_lossP��>�E��       `/�#	Qg��B=�Aj*

val_accȑ2?���x       ��-	h��B=�Aj*

loss���>I�P       ��(	�h��B=�Aj*


acc��=?w���       ��2	)��B=�Ak*

val_lossU�>dAB       `/�#	�*��B=�Ak*

val_acc�2?C�y�       ��-	�+��B=�Ak*

loss���>D"��       ��(	�,��B=�Ak*


acc��=?-=�       ��2	����B=�Al*

val_loss�(�>�R:�       `/�#	ɐ��B=�Al*

val_acc�>8?��l�       ��-	���B=�Al*

loss���>m       ��(	d���B=�Al*


acc�>?t�]l       ��2	U���B=�Am*

val_loss���>3m�y       `/�#	;²�B=�Am*

val_acc�5&?�Ͽ       ��-	�²�B=�Am*

loss,��>}�       ��(	�²�B=�Am*


acc�:?�k�        ��2	2W��B=�An*

val_loss�8?.FW�       `/�#	)Y��B=�An*

val_accy�*?~�U       ��-	�Y��B=�An*

loss!�>!���       ��(	CZ��B=�An*


acc�8?���"       ��2	����B=�Ao*

val_loss���>���       `/�#	����B=�Ao*

val_acc`c:?th^�       ��-	;���B=�Ao*

loss3��>���~       ��(	u���B=�Ao*


acc�{>?۲Z       ��2	<h��B=�Ap*

val_loss׀�>��(       `/�#	Qk��B=�Ap*

val_acc��??��N       ��-	Il��B=�Ap*

loss��>q�r�       ��(	m��B=�Ap*


acc��C?�sLZ       ��2	�_��B=�Aq*

val_lossh	�>�弹       `/�#	+l��B=�Aq*

val_acc=�9?'ad^       ��-	�m��B=�Aq*

lossj��>e?�Y       ��(	�z��B=�Aq*


acc�F;?Ŧ��       ��2	0(��B=�Ar*

val_loss���>j#F       `/�#	N)��B=�Ar*

val_acc�:?��I       ��-	�)��B=�Ar*

loss���>S^J�       ��(	B*��B=�Ar*


acc��??Q��       ��2	�j	�B=�As*

val_loss�O�>���)       `/�#	�k	�B=�As*

val_acctl>?�I�a       ��-	 l	�B=�As*

lossZj�>�-O       ��(	jl	�B=�As*


acc��??mn�       ��2	HR�B=�At*

val_loss"��>�S/       `/�#	aS�B=�At*

val_acc��>?�9ȑ       ��-	T�B=�At*

loss���>���       ��(	�T�B=�At*


acc^�B?%���       ��2	�S*�B=�Au*

val_lossX�>��       `/�#	�T*�B=�Au*

val_acc��D?MޓJ       ��-	?U*�B=�Au*

loss 4�>��\       ��(	�U*�B=�Au*


acc"@?q8��       ��2	�w<�B=�Av*

val_loss3&�>?���       `/�#	�x<�B=�Av*

val_acc$@?М=�       ��-	y<�B=�Av*

loss���>��/       ��(	hy<�B=�Av*


accғ>?Kϕ       ��2	ڬH�B=�Aw*

val_lossZ ?B<��       `/�#	:�H�B=�Aw*

val_accq2?�ݖ       ��-	�H�B=�Aw*

loss,P�>��b�       ��(	_�H�B=�Aw*


acc�/A?�:�       ��2	�gS�B=�Ax*

val_lossBD�>�sq�       `/�#	�hS�B=�Ax*

val_acc� 9?���       ��-	jiS�B=�Ax*

loss���>p[�+       ��(	�iS�B=�Ax*


acc�>;?���       ��2	�]�B=�Ay*

val_lossܦ�>?��       `/�#	��]�B=�Ay*

val_acc��??���       ��-	]�]�B=�Ay*

loss�U�> �H�       ��(	��]�B=�Ay*


acce�A?��R       ��2	pm�B=�Az*

val_loss$0�>s���       `/�#	�pm�B=�Az*

val_acc\E;?"�;�       ��-	`qm�B=�Az*

lossh�>K�n8       ��(	�qm�B=�Az*


acc��2?���       ��2	���B=�A{*

val_loss�s�>�u��       `/�#	���B=�A{*

val_acc�;E??Oi       ��-	|��B=�A{*

loss+�>�;��       ��(	���B=�A{*


accKC?���R       ��2	W��B=�A|*

val_loss���>r��       `/�#	|��B=�A|*

val_acc�P@?���       ��-		��B=�A|*

loss-��>@&t�       ��(	p	��B=�A|*


accAG?q���       ��2	 ��B=�A}*

val_lossց�>�sL*       `/�#	����B=�A}*

val_acc�H?�E�Y       ��-	j���B=�A}*

loss���>�=       ��(	����B=�A}*


accb�@?	�V       ��2	b,��B=�A~*

val_loss_i�>p��O       `/�#	I-��B=�A~*

val_acc{�<?
���       ��-	�-��B=�A~*

loss�r�>�35       ��(	.��B=�A~*


acc$[C?�޸       ��2	���B=�A*

val_lossI��>#i_p       `/�#	���B=�A*

val_acc@UB?mc�_       ��-	V��B=�A*

lossNw�>���h       ��(	���B=�A*


accGF?��"q       QKD	t���B=�A�*

val_loss�R�>[%��       ��2	����B=�A�*

val_acc6�;?�"�       �	���B=�A�*

loss*��>�8�
       ��-	���B=�A�*


acc*0<?=&�       QKD	�f��B=�A�*

val_loss���>"Tt�       ��2	�g��B=�A�*

val_accѕB?�:i       �	8h��B=�A�*

loss=��>�\�       ��-	�h��B=�A�*


acc�OF?W�+�       QKD	/N��B=�A�*

val_loss�%�>�sA�       ��2	YO��B=�A�*

val_accC:?�z�       �	�O��B=�A�*

loss���>��T�       ��-	UP��B=�A�*


acc>E?��I       QKD	Q��B=�A�*

val_lossA] ?���       ��2	�Q��B=�A�*

val_acc�3?���       �	aR��B=�A�*

loss/+�>hb�       ��-	�R��B=�A�*


acck�B?90��       QKD	�C��B=�A�*

val_loss�*�>���       ��2	$E��B=�A�*

val_accQ@G?-w�       �	�E��B=�A�*

lossr|�>Z�d�       ��-	�E��B=�A�*


acc�??��u�       QKD	�Q�B=�A�*

val_lossD4�>L��       ��2	�R�B=�A�*

val_acc�L?���       �	S�B=�A�*

lossR��>����       ��-	�S�B=�A�*


accfqG?��Rh       QKD	,��B=�A�*

val_loss6��>z �9       ��2	��B=�A�*

val_acc.�F??��       �	���B=�A�*

loss+n�>LϪ�       ��-	��B=�A�*


acc�QB?IDo�       QKD	�5"�B=�A�*

val_losspx�><�\4       ��2	�9"�B=�A�*

val_acc��A?�df�       �	�:"�B=�A�*

lossƞ�>��H       ��-	q;"�B=�A�*


accqJ?�d�       QKD	ý/�B=�A�*

val_lossɋ�>EH�       ��2	Ǿ/�B=�A�*

val_acc�I=?�P�a       �	n�/�B=�A�*

loss���>/�A�       ��-	ܿ/�B=�A�*


acc�tD?�==�       QKD	K�;�B=�A�*

val_loss�>�g.2       ��2	G�;�B=�A�*

val_accm0@?��V]       �	��;�B=�A�*

lossp�>��p�       ��-	�;�B=�A�*


acc�<?N�.�       QKD	 �J�B=�A�*

val_loss3��>I�       ��2	9�J�B=�A�*

val_accT^F?�?�V       �	��J�B=�A�*

loss�>�E       ��-	�J�B=�A�*


acc��D?y>
�       QKD	��X�B=�A�*

val_loss��>��l!       ��2	
�X�B=�A�*

val_acc<7C?�Qp       �	��X�B=�A�*

loss���>��?W       ��-	�X�B=�A�*


acc�WF?SVH       QKD	�f�B=�A�*

val_loss���>U��       ��2	� f�B=�A�*

val_acc��>?I-^       �	t!f�B=�A�*

loss�>�I��       ��-	�!f�B=�A�*


acc�yB?�y��       QKD	�Kr�B=�A�*

val_lossE
�>@��       ��2	]Mr�B=�A�*

val_acc�bH? 2��       �	3Nr�B=�A�*

loss�i�>�+��       ��-	�Nr�B=�A�*


accQnE?��	       QKD	�}�B=�A�*

val_loss���>���       ��2	��}�B=�A�*

val_accJ�??k��w       �	C�}�B=�A�*

lossP��> ��       ��-	��}�B=�A�*


acc��C?ۨ�       QKD	)��B=�A�*

val_lossx�>+�w�       ��2	,*��B=�A�*

val_acc@UB?z ��       �	�*��B=�A�*

loss΋�>Z�N       ��-	+��B=�A�*


acc�C?�T�       QKD	�9��B=�A�*

val_loss��>�/��       ��2	�<��B=�A�*

val_acc�n??U�?�       �	p=��B=�A�*

loss��>m�h       ��-	>@��B=�A�*


acck�B?��b       QKD	!���B=�A�*

val_loss���>��G�       ��2	���B=�A�*

val_acc<�L?M9�       �	����B=�A�*

lossQ��>�?$       ��-	ݲ��B=�A�*


acc��G?� N       QKD	����B=�A�*

val_loss|�>yl��       ��2	τ��B=�A�*

val_acc)�5?����       �	w���B=�A�*

loss;b�>#�       ��-	腰�B=�A�*


acctAB?�KF       QKD	����B=�A�*

val_lossۜ�>��L�       ��2	|���B=�A�*

val_acc�n??ŭ��       �	���B=�A�*

loss'��>�z-�       ��-	����B=�A�*


acc�s>?"hy�       QKD	����B=�A�*

val_loss"%�>MuAM       ��2	����B=�A�*

val_acc2�E?�Ad�       �	"���B=�A�*

lossN��>BGPd       ��-	{���B=�A�*


acc>�F?ZPT�       QKD	&W��B=�A�*

val_lossj��>��gg       ��2	�X��B=�A�*

val_acc{�<?Oh��       �	Y��B=�A�*

loss.i�>�z�.       ��-	�Y��B=�A�*


acc�?F?���K       QKD	�-��B=�A�*

val_loss���>d.�       ��2	�.��B=�A�*

val_acc\E?l5       �	Z/��B=�A�*

lossI��>	��#       ��-	�/��B=�A�*


acc�5E?��Ѧ       QKD	��B=�A�*

val_loss5v�>���        ��2	q��B=�A�*

val_accY'<?ͳW       �	���B=�A�*

loss���>|�'       ��-	S��B=�A�*


acc�B?��S�       QKD	�� �B=�A�*

val_loss^ ?j���       ��2	�� �B=�A�*

val_accd,0?�Lf�       �	� �B=�A�*

loss�"�>]Z��       ��-	h� �B=�A�*


acc�<?�>O       QKD	��B=�A�*

val_lossc��>_�6       ��2	��B=�A�*

val_acc�RA?H��       �	!�B=�A�*

loss��>�X�       ��-	��B=�A�*


acc��??� f�       QKD	��B=�A�*

val_loss��>�~��       ��2	���B=�A�*

val_acc G?*.\�       �	M��B=�A�*

loss���>���>       ��-	���B=�A�*


acc�_F?��D       QKD	�6-�B=�A�*

val_loss!��>���       ��2	h<-�B=�A�*

val_acc�YD?ޜ�D       �	C>-�B=�A�*

loss���>��T�       ��-	�?-�B=�A�*


acc��G?��T       QKD	��:�B=�A�*

val_lossO(�>e�|�       ��2	��:�B=�A�*

val_acc�B?����       �	 �:�B=�A�*

loss��>�t       ��-	X�:�B=�A�*


acc<�K?�e       QKD	?!F�B=�A�*

val_lossR�>`B��       ��2	%"F�B=�A�*

val_acc<7C?1��       �	�"F�B=�A�*

loss�=�>Mo�       ��-	�"F�B=�A�*


accV�@?�E�,       QKD	� R�B=�A�*

val_loss���>��ԅ       ��2	�!R�B=�A�*

val_acc(.??[�       �	:"R�B=�A�*

loss���>�Ɉ�       ��-	�"R�B=�A�*


acc�I?��s�       QKD	F�b�B=�A�*

val_loss�U�>}H)       ��2	��b�B=�A�*

val_acc-P?d��z       �	��b�B=�A�*

loss�-�>cU�       ��-	Z�b�B=�A�*


acc�|D?�cT@       QKD	Sn�B=�A�*

val_loss�o�>��oY       ��2	"Tn�B=�A�*

val_accw�F?..       �	�Tn�B=�A�*

loss}��>�zi       ��-	Un�B=�A�*


acc�2C?��>       QKD	q9}�B=�A�*

val_loss�6�>]��       ��2	�:}�B=�A�*

val_accx�=?�K�R       �	.;}�B=�A�*

loss�M�>
��Y       ��-	�;}�B=�A�*


acc_�N?�#�       QKD	�,��B=�A�*

val_loss7��>�%yg       ��2	8.��B=�A�*

val_acc.�F?�u��       �	�.��B=�A�*

lossh{�>l�       ��-	#/��B=�A�*


acc��D?�ݫ       QKD	Un��B=�A�*

val_loss��>���       ��2	To��B=�A�*

val_acc�WC?(:�o       �	�o��B=�A�*

lossy��>����       ��-	p��B=�A�*


acc͟K?����       QKD	֨��B=�A�*

val_lossƈ�>�˶�       ��2	ު��B=�A�*

val_acc��??��       �	����B=�A�*

loss}��>�Ns(       ��-	3���B=�A�*


acc[�J?�1�-       QKD	�b��B=�A�*

val_loss�/�>M���       ��2	�c��B=�A�*

val_acc�n??C'̭       �	d��B=�A�*

loss�e�>��       ��-	od��B=�A�*


acc�?F?���       QKD	c��B=�A�*

val_loss�Y�>���+       ��2	���B=�A�*

val_accY'<?��       �	^��B=�A�*

losso��>�'?�       ��-	���B=�A�*


acc@xA?>5ă       QKD	����B=�A�*

val_lossJq�>`�       ��2		���B=�A�*

val_acc�L0?���c       �	����B=�A�*

lossr��>:3p�       ��-	����B=�A�*


acc�:C?d/e�       QKD	7���B=�A�*

val_loss*��>L�a�       ��2	����B=�A�*

val_acc9D?� @�       �	z ��B=�A�*

loss�u�> [pA       ��-	� ��B=�A�*


acc�?F?���;       QKD	&���B=�A�*

val_loss:��>�zU       ��2	u���B=�A�*

val_acc��<?욳       �	:���B=�A�*

loss�Y�>�I�=       ��-	ސ��B=�A�*


accѬI?m(�       QKD	Nb��B=�A�*

val_lossM�>��#/       ��2	wc��B=�A�*

val_acc��9?��:Y       �	�c��B=�A�*

loss���>�S       ��-	=d��B=�A�*


accg�J?P���       QKD	�?�B=�A�*

val_loss�[�>C�       ��2	B�B=�A�*

val_acc<�L?z�p       �	�B�B=�A�*

loss�^�>[V��       ��-	C�B=�A�*


acc�*C?n�%       QKD	��B=�A�*

val_loss@�>�b=       ��2	c�B=�A�*

val_acc��I?[Ĉ       �	�B=�A�*

loss��>���h       ��-	��B=�A�*


acc��N?��V`       QKD	J&�B=�A�*

val_loss���>���       ��2	@K&�B=�A�*

val_accCsA?\H�       �	#L&�B=�A�*

lossko�>�͗       ��-	�L&�B=�A�*


acc<�K?;��       QKD	�O2�B=�A�*

val_loss}��>9��h       ��2	�P2�B=�A�*

val_acc�C?/f       �	#Q2�B=�A�*

loss��>P��       ��-	�Q2�B=�A�*


acc��C?����       QKD	�?�B=�A�*

val_lossʘ�>�MNB       ��2	?�B=�A�*

val_acc�C?�]{        �	�?�B=�A�*

lossj�>�k�       ��-	?�B=�A�*


acc}<D?\���       QKD	�IQ�B=�A�*

val_lossI&�>48�       ��2	�JQ�B=�A�*

val_acc(.??9
u�       �	�KQ�B=�A�*

loss���> �       ��-	
LQ�B=�A�*


acc��C?J{       QKD	\`�B=�A�*

val_lossWb�>P[�D       ��2	�`�B=�A�*

val_acc��K?�'�s       �	`�B=�A�*

lossA�>m���       ��-	d`�B=�A�*


acc>�F?�B�       QKD	�lm�B=�A�*

val_loss���>��$D       ��2	�mm�B=�A�*

val_acc�MM?Gˍ       �	'nm�B=�A�*

loss���>t	\       ��-	nm�B=�A�*


acc�%J?_JT�       QKD	z�z�B=�A�*

val_loss��>����       ��2	q�z�B=�A�*

val_acc-�+?p��       �	��z�B=�A�*

lossW�>�6        ��-	\�z�B=�A�*


accj~E?�+w$       QKD	���B=�A�*

val_lossw�>|��       ��2	���B=�A�*

val_accigJ?�8ؽ       �	��B=�A�*

loss+�>;~�A       ��-	^��B=�A�*


acc�IB?f��       QKD	ϡ��B=�A�*

val_loss%�>Y��       ��2	����B=�A�*

val_acc�U4?֡�p       �	/���B=�A�*

lossց�>�O�p       ��-	����B=�A�*


acc��M?&B�u       QKD	�V��B=�A�*

val_lossG��>��
�       ��2	%X��B=�A�*

val_acc G?`���       �	�X��B=�A�*

loss!��>#X;�       ��-	&Y��B=�A�*


acc��H?dl}Z       QKD	�ط�B=�A�*

val_loss���>��	       ��2	�ٷ�B=�A�*

val_accL?\���       �	#ڷ�B=�A�*

loss ��>���       ��-	�ڷ�B=�A�*


acc*�J?`Y�       QKD	k��B=�A�*

val_loss��>�n�       ��2	�l��B=�A�*

val_acc�I=?��7�       �	7m��B=�A�*

loss��>7��       ��-	�n��B=�A�*


acc(IG?��C       QKD	����B=�A�*

val_lossl9�>���       ��2	���B=�A�*

val_acc��0?�>�       �	����B=�A�*

loss��>q;��       ��-	���B=�A�*


acc <?~��       QKD	g���B=�A�*

val_loss��>!!�        ��2	'���B=�A�*

val_acc�3?���       �	����B=�A�*

loss��>���       ��-	K��B=�A�*


accK;>?jЁM       QKD	@���B=�A�*

val_lossK^�>��|�       ��2	����B=�A�*

val_acc��K?�@ٚ       �	T���B=�A�*

loss��>0.f�       ��-	����B=�A�*


acc��L?b냷       QKD	~���B=�A�*

val_lossۍ�>T��       ��2	����B=�A�*

val_acc�TP?�@�x       �	8���B=�A�*

loss�F�>rm�v       ��-	����B=�A�*


accmL?~�u�       QKD	��B=�A�*

val_loss?�>
��       ��2	מ�B=�A�*

val_acc�E?��ņ       �	Y��B=�A�*

loss���>X�|       ��-	ß�B=�A�*


acc}<D?���Z       QKD	1��B=�A�*

val_loss_��>�D{�       ��2	,��B=�A�*

val_acc޷Q?�|��       �	���B=�A�*

loss���>�X�k       ��-	��B=�A�*


acc#�K?S]�       QKD	�-�B=�A�*

val_lossz��>qK�       ��2	�-�B=�A�*

val_acc�N?�G+}       �	2-�B=�A�*

lossT�>��       ��-	�-�B=�A�*


accoMO?2�{�       QKD	�K�B=�A�*

val_loss��>p��       ��2	�	K�B=�A�*

val_acc�MM?�6T�       �	)
K�B=�A�*

loss���>0�iY       ��-	�
K�B=�A�*


acc�NP?L��       QKD	;[�B=�A�*

val_loss'�>6��       ��2	m[�B=�A�*

val_acc�F?�d�[       �	�[�B=�A�*

loss��>�|�       ��-	`[�B=�A�*


acc�0L?�x3Z       QKD	��f�B=�A�*

val_lossk��>�l�0       ��2	<�f�B=�A�*

val_acc��5?d!        �	��f�B=�A�*

loss�/�>5�O       ��-	�f�B=�A�*


acc�BC?|i�       QKD	��s�B=�A�*

val_loss��>��&�       ��2	r t�B=�A�*

val_acc�-M?s�VJ       �	�t�B=�A�*

loss�8�>��}T       ��-	�t�B=�A�*


acc��G?�w.*       QKD	����B=�A�*

val_lossm��>���       ��2	����B=�A�*

val_acc<7C?6`�+       �	(���B=�A�*

lossÛ�>c�L       ��-	����B=�A�*


acc}J?% ��       QKD	�ٍ�B=�A�*

val_loss���>���       ��2	0ۍ�B=�A�*

val_acc[�D?��!       �	�ۍ�B=�A�*

lossF��>].�       ��-	܍�B=�A�*


accA:M?�t��       QKD	Ж��B=�A�*

val_loss���>4�@b       ��2	���B=�A�*

val_acc��??}8�       �	g���B=�A�*

loss�>��       ��-	1���B=�A�*


accj_K?��r       QKD	9���B=�A�*

val_loss���>�?       ��2	-���B=�A�*

val_acc GJ?����       �	����B=�A�*

loss�V�>���       ��-	����B=�A�*


acc�\D?��       QKD	�Ʒ�B=�A�*

val_loss�d�>��       ��2	�Ƿ�B=�A�*

val_acc�RA?v        �	?ȷ�B=�A�*

loss��>*0 s       ��-	�ȷ�B=�A�*


accEGK?��J�       QKD	M��B=�A�*

val_loss���>i� �       ��2	�M��B=�A�*

val_acc{�<?�ș�       �	fN��B=�A�*

loss��>��ܬ       ��-	�N��B=�A�*


accvJ?ﶁ�       QKD	?��B=�A�*

val_loss]}�>^$G6       ��2	�@��B=�A�*

val_accG�@?� *       �	A��B=�A�*

lossR�>D͚4       ��-	lA��B=�A�*


acc��=?�D��       QKD	���B=�A�*

val_loss�X�>_�_t       ��2	��B=�A�*

val_acc��K?QY��       �	o��B=�A�*

loss���>l�       ��-	���B=�A�*


acc��J?��f       QKD	V)��B=�A�*

val_loss��>&΃z       ��2	
+��B=�A�*

val_acc�;?�ɦ�       �	�+��B=�A�*

loss��>eȓ       ��-	,��B=�A�*


acc`F?�kL�       QKD	:���B=�A�*

val_loss���>jZ�       ��2	���B=�A�*

val_acc��J?���       �	����B=�A�*

loss���>]/(#       ��-	����B=�A�*


acc
�K?'�r       QKD	���B=�A�*

val_lossm��>����       ��2	���B=�A�*

val_acc��K?&�n6       �	��B=�A�*

loss܍�>T&��       ��-	g��B=�A�*


acc~J?�K�.       QKD	�:�B=�A�*

val_loss��>][��       ��2	�;�B=�A�*

val_acc�C?y~`�       �	:<�B=�A�*

loss�n�>��O       ��-	�<�B=�A�*


acc��E?q��       QKD	��B=�A�*

val_loss��><�y�       ��2	��B=�A�*

val_accWQ?BY�>       �	B�B=�A�*

loss���>�J.�       ��-	��B=�A�*


acc#�K?�~       QKD	r3,�B=�A�*

val_lossu��>DWE       ��2	j4,�B=�A�*

val_acc�C?dH\�       �	�4,�B=�A�*

loss�o�>
�Fg       ��-	H5,�B=�A�*


acc��P?�3��       QKD	I�8�B=�A�*

val_loss��>�_�       ��2	��8�B=�A�*

val_acciA?�xB       �	#�8�B=�A�*

loss�>b��       ��-	��8�B=�A�*


acc�:H?�qR�       QKD	g	G�B=�A�*

val_loss-��>���o       ��2	[
G�B=�A�*

val_acc��9?�hϵ       �	�
G�B=�A�*

loss4��>��\       ��-	BG�B=�A�*


acc/�K?���I       QKD	�2U�B=�A�*

val_lossS��>>>�0       ��2	�3U�B=�A�*

val_acc�DI?E��b       �	34U�B=�A�*

loss�*�>��       ��-	�4U�B=�A�*


accxN@?��lu       QKD	��b�B=�A�*

val_loss='�>�v��       ��2	�b�B=�A�*

val_acc��P?����       �	��b�B=�A�*

loss��>=��V       ��-	�b�B=�A�*


acc2�L?�       QKD	͎p�B=�A�*

val_loss�~�>��N"       ��2	��p�B=�A�*

val_acc�YD?K�       �	*�p�B=�A�*

loss���>=Jb�       ��-	��p�B=�A�*


acc�PQ?T�v       QKD	�!~�B=�A�*

val_loss�B�>h�       ��2	�"~�B=�A�*

val_acc��C?ރ�       �	-#~�B=�A�*

loss���>�w��       ��-	�#~�B=�A�*


acc�rM?=�K       QKD	���B=�A�*

val_losseS�>����       ��2	 ��B=�A�*

val_acc�+>?.�s�       �	� ��B=�A�*

loss��><�/�       ��-	� ��B=�A�*


acc"�E?���       QKD	]��B=�A�*

val_loss���>s\�       ��2	f��B=�A�*

val_acc��@?�ZÆ       �	���B=�A�*

loss�S�>�G?�       ��-	T��B=�A�*


acc2�L?D4�       QKD	�G��B=�A�*

val_loss��>�>�       ��2	HK��B=�A�*

val_accN�>? �8�       �	@M��B=�A�*

loss?��>���       ��-	�N��B=�A�*


acc��F?�DZ       QKD	��B=�A�*

val_lossTU�>�       ��2	���B=�A�*

val_acc�C?O��7       �	9���B=�A�*

losso��>Įb       ��-	o��B=�A�*


accztI?�HY       QKD	����B=�A�*

val_loss ��>Tk�       ��2	���B=�A�*

val_acc�<?�T��       �	���B=�A�*

loss!t�>y4�       ��-	���B=�A�*


accEGK?��       QKD	�c��B=�A�*

val_loss��>�ն       ��2	td��B=�A�*

val_accb�B?@�'k       �	�d��B=�A�*

loss���>(�'�       ��-	Ie��B=�A�*


acc��E?�~�       QKD	u���B=�A�*

val_loss[=�>\k,w       ��2	����B=�A�*

val_acc��.?��,       �	?���B=�A�*

loss��>�m�X       ��-	����B=�A�*


accG�7?�;Z       QKD	4���B=�A�*

val_lossp�>'��       ��2	D���B=�A�*

val_accigJ? ���       �	����B=�A�*

loss�&�>X���       ��-	D���B=�A�*


acc=LI?����       QKD	+o��B=�A�*

val_loss���>*b�       ��2	�r��B=�A�*

val_accM"H?i�*�       �	U���B=�A�*

lossX�>�V�       ��-	z���B=�A�*


acc�N?�A��       QKD	�B=�A�*

val_lossN��>"��^       ��2	_�B=�A�*

val_acc�N?�h��       �	��B=�A�*

lossl��>�7A       ��-	E�B=�A�*


accX�R?�l��       QKD	p��B=�A�*

val_loss��>�Რ       ��2	l��B=�A�*

val_acc�p@?`�t>       �	��B=�A�*

loss+�>}�       ��-	}��B=�A�*


acc�GF?����       QKD	5
)�B=�A�*

val_loss���>��       ��2	J)�B=�A�*

val_acc��J?�d�       �	�)�B=�A�*

loss�Y�>!��	       ��-	J)�B=�A�*


acc��O?v��       QKD	�K:�B=�A�*

val_loss���>�a�       ��2	U:�B=�A�*

val_acc�B?8Bb�       �	�X:�B=�A�*

loss9��>��}       ��-	�[:�B=�A�*


acci�M?�w�E       QKD	�&H�B=�A�*

val_loss��>&�8       ��2	(H�B=�A�*

val_acc��C?J\[�       �	�(H�B=�A�*

loss ��>�q�q       ��-	c*H�B=�A�*


acc�BH?y��o       QKD	��T�B=�A�*

val_loss.w�>�       ��2	��T�B=�A�*

val_acc�$I?֑&V       �	I�T�B=�A�*

loss��>��i       ��-	��T�B=�A�*


acclN?��{�       QKD	:�b�B=�A�*

val_lossC��>u�       ��2	��b�B=�A�*

val_accճA?���       �	��b�B=�A�*

loss��>3�[�       ��-	X�b�B=�A�*


accT�K?s��T       QKD	�bn�B=�A�*

val_loss���>Z&:�       ��2	�cn�B=�A�*

val_accC�J?8�F�       �	sdn�B=�A�*

lossޱ�>?r�       ��-	�dn�B=�A�*


acc�E?�m��       QKD	h@}�B=�A�*

val_loss���>�qO�       ��2	�A}�B=�A�*

val_acc��J??l�       �	B}�B=�A�*

loss��>�vc�       ��-	tB}�B=�A�*


acc\H?��M�       QKD	���B=�A�*

val_loss�D�>�
       ��2	�	��B=�A�*

val_acc<7C?�7)�       �	>
��B=�A�*

lossT�>�̕       ��-	�
��B=�A�*


acc��J?���\       QKD	�O��B=�A�*

val_losscx�>���       ��2	P��B=�A�*

val_acc��M?���       �	�P��B=�A�*

loss5��>W�+       ��-	TQ��B=�A�*


acc}J?率�       QKD	�_��B=�A�*

val_lossI�><s��       ��2	�`��B=�A�*

val_acc[�M?6�       �	(a��B=�A�*

loss���>��9D       ��-	�a��B=�A�*


acc�(L?c�c�       QKD	���B=�A�*

val_loss1T�>Ғ {       ��2	}��B=�A�*

val_accz�E?��$       �	��B=�A�*

loss`C�>����       ��-	x��B=�A�*


acc_�N?,��l       QKD	g��B=�A�*

val_lossDz�>���E       ��2	k��B=�A�*

val_acc��I?T�p       �	�l��B=�A�*

lossy�>X�x�       ��-	�n��B=�A�*


acc�8L?��M       QKD	V���B=�A�*

val_loss��>ڮh�       ��2	���B=�A�*

val_acc��O?�Iߨ       �	���B=�A�*

loss���>�_��       ��-	I���B=�A�*


acc�_F?`���       QKD	Z���B=�A�*

val_losso7�>�nP       ��2	R���B=�A�*

val_acc GJ?jI       �	ǝ��B=�A�*

loss%��>�z�       ��-	M���B=�A�*


accɒM?B���       QKD	Z���B=�A�*

val_loss���>QRb�       ��2	M���B=�A�*

val_acc2�E?��K�       �	˻��B=�A�*

loss��>����       ��-	0���B=�A�*


acc�O??��       QKD	3���B=�A�*

val_loss�Q�>�'��       ��2	����B=�A�*

val_acc<�L?[ѯ8       �	a���B=�A�*

loss���>�c�g       ��-	����B=�A�*


acc�+4?�Ǆ�       QKD	c���B=�A�*

val_loss�?�>��A$       ��2	����B=�A�*

val_acc��F?t Y�       �	=���B=�A�*

loss�=�>IA�'       ��-	����B=�A�*


acc�]O?�[>       QKD	r�B=�A�*

val_loss�;�>�?��       ��2	!s�B=�A�*

val_acc��O?bܙ       �	�s�B=�A�*

loss'�>dlc�       ��-	;t�B=�A�*


acc�rM?I�ֹ       QKD	 ��B=�A�*

val_loss�1�>�>�       ��2	N��B=�A�*

val_accɮM?᳡       �	���B=�A�*

loss{�>ǐ       ��-	��B=�A�*


acc�O?�!�       QKD	?�B=�A�*

val_loss:��>�e�       ��2	�@�B=�A�*

val_accɮM?�xYl       �	�A�B=�A�*

lossy�>��%�       ��-	B�B=�A�*


acc��N?� �       QKD	t�)�B=�A�*

val_losswN�>mg>N       ��2	l�)�B=�A�*

val_acc̯1?��b�       �	��)�B=�A�*

loss�w�>�)�       ��-	l�)�B=�A�*


acc��L?�(Ts       QKD	w�6�B=�A�*

val_lossÌ�>O�       ��2	�6�B=�A�*

val_acc�BH?t�y�       �	�6�B=�A�*

loss�<�>��-m       ��-	r�6�B=�A�*


acc"�E?�e�Q       QKD	u�H�B=�A�*

val_loss&��>�>       ��2	��H�B=�A�*

val_acc��S?���J       �	��H�B=�A�*

loss�6�>'��       ��-	��H�B=�A�*


acc�Q?ؾ�A       QKD	<V�B=�A�*

val_lossk��>�8o       ��2	=V�B=�A�*

val_accC�J?�U7       �	�=V�B=�A�*

loss@C�>5�9       ��-	 >V�B=�A�*


acc�O?�b�       QKD	�]b�B=�A�*

val_loss���>p&D+       ��2	F_b�B=�A�*

val_acc�G<?Q���       �	�_b�B=�A�*

loss|�>f�q�       ��-	1`b�B=�A�*


acc�K?���       QKD	��o�B=�A�*

val_loss��>Aj�       ��2	Ӂo�B=�A�*

val_acc[�M?c^��       �	Y�o�B=�A�*

loss+�>R'�"       ��-	�o�B=�A�*


acc/�K?-��u       QKD	���B=�A�*

val_lossR��>�=�z       ��2	���B=�A�*

val_acc��E?��       �	l��B=�A�*

lossX��>��Q�       ��-	���B=�A�*


acc"M?�&�       QKD	<���B=�A�*

val_loss���>�}7       ��2	+���B=�A�*

val_acc+L>?F��       �	����B=�A�*

lossP��>�aҁ       ��-	/���B=�A�*


acc7B?���~       QKD	����B=�A�*

val_loss�O�>u�v}       ��2	��B=�A�*

val_accp�H?���       �	c���B=�A�*

loss�}�>I���       ��-	����B=�A�*


acc�#I?�e{       QKD	����B=�A�*

val_loss!��> V�N       ��2	r���B=�A�*

val_acc��C?�x�       �	6���B=�A�*

loss�>�si       ��-	����B=�A�*


acc	dN?w6z       QKD	���B=�A�*

val_lossW��>���<       ��2	���B=�A�*

val_acc�M?�$��       �	K��B=�A�*

loss���>k       ��-	���B=�A�*


acc��H?k#{       QKD	���B=�A�*

val_lossH��>����       ��2	B ��B=�A�*

val_accճA?g��d       �	�!��B=�A�*

loss�B�>t1P�       ��-	B"��B=�A�*


acc*�J?���       QKD	����B=�A�*

val_lossf��>���       ��2	����B=�A�*

val_acc��F?* e       �	����B=�A�*

loss���>��P2       ��-	����B=�A�*


acct�J?T�Ll       QKD	ض��B=�A�*

val_loss)��>�A��       ��2	���B=�A�*

val_acc_�C?��_       �	o���B=�A�*

lossZ�>��w       ��-	Ǹ��B=�A�*


acc&D?Z�6�       QKD	�
��B=�A�*

val_loss���>��y       ��2	=��B=�A�*

val_accb+L?��&       �	9��B=�A�*

lossW��>�T��       ��-	���B=�A�*


acc)*M?���       QKD	�J �B=�A�*

val_loss5��>a��       ��2	�K �B=�A�*

val_accƐN?�&ȑ       �	QL �B=�A�*

loss���>�IPg       ��-	�L �B=�A�*


acc�HL?l��       QKD	��	�B=�A�*

val_lossz)�>��S'       ��2	ϥ	�B=�A�*

val_acc��I?m�be       �	2�	�B=�A�*

loss���>���       ��-	q�	�B=�A�*


accwHQ? u�Q       QKD	{��B=�A�*

val_loss���>/"�       ��2	���B=�A�*

val_accH<7?��       �	Y��B=�A�*

loss�/�>]�       ��-	���B=�A�*


acc�B?ҵ�       QKD	�\&�B=�A�*

val_lossD+�>~��       ��2	>^&�B=�A�*

val_acczD?�&�a       �	�^&�B=�A�*

lossǱ�>�       ��-	1_&�B=�A�*


acc�B?Y���       QKD	��0�B=�A�*

val_loss�S�>*��       ��2	��0�B=�A�*

val_acc��U?ڏ��       �	T�0�B=�A�*

loss�U�>3Ϥj       ��-	��0�B=�A�*


accO�P?Q�'�       QKD	��<�B=�A�*

val_loss���>�BZ	       ��2	C�<�B=�A�*

val_acc}pN? '       �	Ί<�B=�A�*

loss��>�(U�       ��-	@�<�B=�A�*


acc�P?�ɟ!       QKD	K�B=�A�*

val_loss�P�>�� X       ��2	�K�B=�A�*

val_acc��J?�`��       �	EK�B=�A�*

lossA��>���       ��-	�K�B=�A�*


acc�CN?��V�       QKD	�W�B=�A�*

val_loss��>Np$       ��2	["W�B=�A�*

val_acch�S?�@�       �	)#W�B=�A�*

lossw��>���       ��-	�#W�B=�A�*


accwHQ?*<w2       QKD	�2c�B=�A�*

val_loss��>;��'       ��2	�3c�B=�A�*

val_acc>F?b�hy       �	/4c�B=�A�*

loss���>Tc�       ��-	�4c�B=�A�*


accMaG?5~x       QKD	ۤp�B=�A�*

val_lossM��>@pe�       ��2	��p�B=�A�*

val_accC�J?w���       �	?�p�B=�A�*

loss�`�>�rf�       ��-	��p�B=�A�*


acc��N?��9       QKD	U�~�B=�A�*

val_loss���>V�        ��2	L�~�B=�A�*

val_accBT?��+       �	Ƨ~�B=�A�*

loss���>��܁       ��-	'�~�B=�A�*


acc<�K?�A�       QKD	���B=�A�*

val_lossP��>����       ��2	���B=�A�*

val_acco�Q?�U$       �	��B=�A�*

lossA��>����       ��-	|��B=�A�*


acc�I?��>       QKD	֋��B=�A�*

val_loss���>~�s+       ��2	댕�B=�A�*

val_acctl>?n~)�       �	����B=�A�*

lossU��>���       ��-	�B=�A�*


acc��L?�wH�       QKD	���B=�A�*

val_loss
��>�Ů�       ��2	��B=�A�*

val_acc��B?�vZ�       �	���B=�A�*

loss�H�>��H       ��-	���B=�A�*


acc_�N?/��       QKD	���B=�A�*

val_loss+;�>�P�       ��2	`��B=�A�*

val_acc�B?!a��       �	���B=�A�*

loss���>z��       ��-	S��B=�A�*


accMaG?�Z7M       QKD	u>��B=�A�*

val_loss��>j�
       ��2	d?��B=�A�*

val_acc�B?ǖA�       �	�?��B=�A�*

lossT��>�;�       ��-	6@��B=�A�*


acc�C?e�o/       QKD	.���B=�A�*

val_loss��>��       ��2	����B=�A�*

val_acc��A?<���       �	T���B=�A�*

loss���>��       ��-	����B=�A�*


acc�5J?�ʽ       QKD	;��B=�A�*

val_lossR��>:$I{       ��2	<��B=�A�*

val_accH?R�J7       �	�<��B=�A�*

loss��>#�z-       ��-	=��B=�A�*


acc٧K?��<       QKD	_���B=�A�*

val_loss)0�>��       ��2	����B=�A�*

val_acck�R?�%�-       �	���B=�A�*

loss���>/ľ4       ��-	[���B=�A�*


accL�O?B��       QKD	����B=�A�*

val_loss��>�}B       ��2	1���B=�A�*

val_accJI?->.�       �	����B=�A�*

lossP��>V�       ��-	���B=�A�*


acc�HL?oRs�       QKD	���B=�A�*

val_loss���>dT�       ��2	&���B=�A�*

val_acc�$I?R�TI       �	����B=�A�*

loss���>��KE       ��-	���B=�A�*


acco�F?n���       QKD	����B=�A�*

val_loss��>e�&�       ��2	���B=�A�*

val_accȑ2?�d%       �	����B=�A�*

lossL�>G���       ��-	����B=�A�*


accbEO?3�       QKD	f��B=�A�*

val_loss���>g�˞       ��2	
��B=�A�*

val_acc��N?I�A�       �	��B=�A�*

lossX��>��q�       ��-	ۊ�B=�A�*


acc~J? e�~       QKD	���B=�A�*

val_loss�o�>ߙt       ��2	���B=�A�*

val_acc�E?��r       �	[��B=�A�*

loss���>|��3       ��-	b��B=�A�*


acc�oF?\�c       QKD	�=$�B=�A�*

val_loss7��>dG#:       ��2	�>$�B=�A�*

val_acc�2?�v�       �	�?$�B=�A�*

loss���>Ҷ��       ��-	�?$�B=�A�*


acc��G?X8\�       QKD	Wy0�B=�A�*

val_loss+O�>��KI       ��2	cz0�B=�A�*

val_accƐN?~G�g       �	�z0�B=�A�*

lossp�>��-$       ��-	S{0�B=�A�*


acc��M?�b�i       QKD	�=�B=�A�*

val_loss��>�d�       ��2	.=�B=�A�*

val_acc'�H?���q       �	�=�B=�A�*

loss���>�b<l       ��-	� =�B=�A�*


acc4�O?�i�       QKD	��I�B=�A�*

val_loss�T�>hGk�       ��2	��I�B=�A�*

val_acc��C?�(�       �	��I�B=�A�*

loss�=�>7�O�       ��-	��I�B=�A�*


accG.@?�Uh       QKD	A�V�B=�A�*

val_loss`��>p�\       ��2	0�V�B=�A�*

val_acc��K?��a�       �	��V�B=�A�*

lossm��>�?�       ��-	�V�B=�A�*


acc"M?e�~�       QKD	Jb�B=�A�*

val_loss��>2�qY       ��2	sb�B=�A�*

val_accM"H?
NH       �	�b�B=�A�*

loss%�>����       ��-	=b�B=�A�*


acc�P?�N,       QKD	�m�B=�A�*

val_loss6�>���       ��2	h"m�B=�A�*

val_accL?��"�       �	�#m�B=�A�*

loss9��>����       ��-	[$m�B=�A�*


acc�NP?�       QKD	o�v�B=�A�*

val_loss'��>�y       ��2	b�v�B=�A�*

val_acc��O?'��       �	��v�B=�A�*

loss���>;���       ��-	E�v�B=�A�*


acc��K?�N�@       QKD	�]�B=�A�*

val_lossAX�>T��       ��2	�^�B=�A�*

val_acc�C?�W	       �	V_�B=�A�*

loss�a�>�\`�       ��-	�_�B=�A�*


accG�N?uj�I       QKD	G5��B=�A�*

val_loss���>Ε��       ��2	v6��B=�A�*

val_acc�BH?)u�       �	�6��B=�A�*

loss�S�>eG       ��-	?7��B=�A�*


accU{C?$�ʓ       QKD	��B=�A�*

val_loss�7�>���       ��2	l��B=�A�*

val_acc�J?�L͹       �	���B=�A�*

loss>��>���       ��-	J��B=�A�*


acc��P?t<\       QKD	����B=�A�*

val_lossb��>��G       ��2	����B=�A�*

val_acc`c:?tU،       �	1���B=�A�*

loss�=�>�r�       ��-	����B=�A�*


acc\�P?4�bR       QKD	���B=�A�*

val_losss��>j��        ��2	���B=�A�*

val_acc�02?��b�       �	h��B=�A�*

loss��>��       ��-	���B=�A�*


acco�F?�/       QKD	M��B=�A�*

val_loss�Q�>b݀       ��2	N��B=�A�*

val_acc��J?�#�       �	�N��B=�A�*

loss�w�>Vt       ��-	�N��B=�A�*


acclF@?3�Y