       �K"	  ��B=�Abrain.Event:2cͿ	�     ��$	���B=�A"��
j
input_1Placeholder*
shape:���������+*
dtype0*'
_output_shapes
:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+   
   *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�D��
_
dense_1/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:+
*
seed2끞*

seed
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:+

~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+

�
dense_1/kernel
VariableV2*
shape
:+
*
shared_name *
dtype0*
_output_shapes

:+
*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+

{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+

Z
dense_1/ConstConst*
valueB
*    *
dtype0*
_output_shapes
:

x
dense_1/bias
VariableV2*
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_1/bias
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������

g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������

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
:���������

s
"dense_1/activity_regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
g
"dense_1/activity_regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
 dense_1/activity_regularizer/addAdd"dense_1/activity_regularizer/add/x dense_1/activity_regularizer/Sum*
_output_shapes
: *
T0
m
dense_2/random_uniform/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *��!�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *��!?
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*

seed*
T0*
dtype0*
_output_shapes

:
*
seed2�ߜ
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes

:
*
T0
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes

:
*
T0
�
dense_2/kernel
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:
*
use_locking(
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:

Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_3/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
_
dense_3/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�KF�
_
dense_3/random_uniform/maxConst*
valueB
 *�KF?*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
_output_shapes

:*
seed2��*

seed*
T0*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:*
T0
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:*
T0
�
dense_3/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Z
dense_3/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_3/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
W
dense_3/TanhTanhdense_3/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_4/random_uniform/shapeConst*
valueB"   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *���*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
_output_shapes

:+*
seed2�Ֆ*

seed
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:+
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:+
�
dense_4/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:+*
	container *
shape
:+
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:+
Z
dense_4/ConstConst*
_output_shapes
:+*
valueB+*    *
dtype0
x
dense_4/bias
VariableV2*
dtype0*
_output_shapes
:+*
	container *
shape:+*
shared_name 
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
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
data_formatNHWC*'
_output_shapes
:���������+*
T0
W
dense_4/ReluReludense_4/BiasAdd*
T0*'
_output_shapes
:���������+
_
Adam/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
s
Adam/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(
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
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
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
dtype0*
_output_shapes
: *
	container *
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
Adam/beta_2/readIdentityAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
T0
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
�
dense_4_targetPlaceholder*0
_output_shapes
:������������������*%
shape:������������������*
dtype0
q
dense_4_sample_weightsPlaceholder*#
_output_shapes
:���������*
shape:���������*
dtype0
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
(loss/dense_4_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*
T0*#
_output_shapes
:���������
a
loss/dense_4_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*
T0*#
_output_shapes
:���������
�
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*
Truncate( *#
_output_shapes
:���������*

DstT0*

SrcT0

a
loss/dense_4_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
: *
	keep_dims( *

Tidx0*
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
loss/mul/xloss/dense_4_loss/Mean_3*
_output_shapes
: *
T0
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
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:���������*

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*
valueB:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
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
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/y*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *
T0
�
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0
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
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
_output_shapes

:
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
:���������
*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
_output_shapes
:*
T0*
out_type0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *,
_class"
 loc:@loss/dense_4_loss/truediv
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
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
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
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
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1Shape dense_1/activity_regularizer/Abs*
T0*
out_type0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
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
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������

�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
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
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
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
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*2
_output_shapes 
:���������:���������
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
:���������

�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeConst*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/addAdd*loss/dense_4_loss/Mean_1/reduction_indices:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/startConst*
value	B : *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*
_output_shapes
:*

Tidx0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/valueConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
N*
_output_shapes
:
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
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
>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordivFloorDiv;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2Shapeloss/dense_4_loss/Mean*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:*
T0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
T0*
out_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
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
Truncate( *
_output_shapes
: *

DstT0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeloss/dense_4_loss/Square*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
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
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
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
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*

Tidx0
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*)
_class
loc:@loss/dense_4_loss/Mean
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
N*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*
T0
�
<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
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
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
:
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_4_loss/Mean*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/ConstConst<^training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv*
valueB
 *   @*+
_class!
loc:@loss/dense_4_loss/Square*
dtype0*
_output_shapes
: 
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
8training/Adam/gradients/loss/dense_4_loss/sub_grad/ShapeShapedense_4/Relu*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub*
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
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/sub
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/sub*0
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
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
T0*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_4/MatMul*
_output_shapes

:+*
transpose_a(*
transpose_b( 
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*
_class
loc:@dense_3/Tanh*'
_output_shapes
:���������*
T0
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:���������*
transpose_a( 
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0*"
_class
loc:@dense_2/BiasAdd
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:
*
transpose_a(
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
N*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������

�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:

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
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
_output_shapes

:+
*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_1/MatMul
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( 
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
training/Adam/sub/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
h
training/Adam/zerosConst*
valueB+
*    *
dtype0*
_output_shapes

:+

�
training/Adam/Variable
VariableV2*
dtype0*
_output_shapes

:+
*
	container *
shape
:+
*
shared_name 
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+
*
use_locking(
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+

b
training/Adam/zeros_1Const*
_output_shapes
:
*
valueB
*    *
dtype0
�
training/Adam/Variable_1
VariableV2*
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:

�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:

j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_2
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:
*
use_locking(
�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
_output_shapes

:
*
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_3
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:
j
training/Adam/zeros_4Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_4
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
_output_shapes

:*
T0*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
j
training/Adam/zeros_6Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_6
VariableV2*
dtype0*
_output_shapes

:+*
	container *
shape
:+*
shared_name 
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes

:+*
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
VariableV2*
dtype0*
_output_shapes
:+*
	container *
shape:+*
shared_name 
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
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes
:+*
T0*+
_class!
loc:@training/Adam/Variable_7
j
training/Adam/zeros_8Const*
_output_shapes

:+
*
valueB+
*    *
dtype0
�
training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
_output_shapes

:+
*
	container *
shape
:+

�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
_output_shapes

:+
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
_output_shapes

:+
*
T0*+
_class!
loc:@training/Adam/Variable_8
b
training/Adam/zeros_9Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:

k
training/Adam/zeros_10Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
_output_shapes

:
*
T0*,
_class"
 loc:@training/Adam/Variable_10
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_11
VariableV2*
_output_shapes
:*
	container *
shape:*
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
:
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
k
training/Adam/zeros_12Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_12
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
_output_shapes

:*
T0*,
_class"
 loc:@training/Adam/Variable_12
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_13
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:*
T0
k
training/Adam/zeros_14Const*
dtype0*
_output_shapes

:+*
valueB+*    
�
training/Adam/Variable_14
VariableV2*
dtype0*
_output_shapes

:+*
	container *
shape
:+*
shared_name 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
_output_shapes

:+*
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
VariableV2*
dtype0*
_output_shapes
:+*
	container *
shape:+*
shared_name 
�
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15
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
training/Adam/zeros_16/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_16
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
T0*,
_class"
 loc:@training/Adam/Variable_16*
_output_shapes
:
p
&training/Adam/zeros_17/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_17
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
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
_output_shapes
:*
T0*

index_type0
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
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_18
p
&training/Adam/zeros_19/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_19/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_19
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
:*
T0
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
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_20
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_20
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_22
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
:
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
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_23
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
_output_shapes

:+
*
T0
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

:+

m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+

t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes

:+
*
T0
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
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:+
*
T0
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:+

m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

:+
*
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
_output_shapes

:+
*
T0
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
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*
_output_shapes

:+

�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*
_output_shapes

:+

d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
_output_shapes

:+
*
T0
Z
training/Adam/add_3/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
_output_shapes

:+
*
T0
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes

:+

q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:+

�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+

�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
validate_shape(*
_output_shapes

:+
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+

p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
:
*
T0
Z
training/Adam/sub_5/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:
*
T0
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:

Z
training/Adam/sub_6/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:

i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:

h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes
:
*
T0
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
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
:

�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:

`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:

Z
training/Adam/add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
:

r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:
*
T0
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:

�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense_1/bias
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:

Z
training/Adam/sub_8/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:
*
T0
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:

Z
training/Adam/sub_9/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:
*
T0
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:

l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:

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

:

�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
_output_shapes

:
*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
_output_shapes

:
*
T0
Z
training/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *���3*
dtype0
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes

:
*
T0
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
_output_shapes

:
*
T0
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:

�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
_output_shapes
:*
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
:
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:
[
training/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes
:
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes
:*
T0
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
:
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
:*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:
[
training/Adam/add_12/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes
:
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
_output_shapes
:*
T0
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
_output_shapes

:*
T0
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

:
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:
[
training/Adam/sub_15/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:
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

:
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
_output_shapes

:*
T0
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:
[
training/Adam/add_15/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*
_output_shapes

:
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
_output_shapes

:*
T0
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes
:
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
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:
[
training/Adam/sub_18/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
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
:
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
:
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:
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
:*
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes
:
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:+
[
training/Adam/sub_20/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes

:+*
T0
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:+
[
training/Adam/sub_21/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:+
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes

:+*
T0
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:+
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

:+
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:+
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes

:+*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:+
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

:+*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
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
training/Adam/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:+
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes
:+*
T0
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
training/Adam/Const_17Const*
dtype0*
_output_shapes
: *
valueB
 *  �
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
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
_output_shapes
:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(
�
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15
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
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_3/kernel
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*
_output_shapes
: *!
_class
loc:@dense_4/kernel*
dtype0
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
IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_output_shapes
: *
_class
loc:@Adam/beta_1*
dtype0
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
IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_1
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
IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4*
dtype0
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
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_8*
dtype0
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
IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15
�
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17
�
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_19*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_19*
dtype0
�
IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_20*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_20
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
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"
 ݚ*�     �:S[	���B=�AJ��
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
shape:���������+*
dtype0*'
_output_shapes
:���������+
m
dense_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"+   
   
_
dense_1/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
seed2끞*
_output_shapes

:+
*

seed
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:+

~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+

�
dense_1/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:+
*
shape
:+
*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+
*
use_locking(*
T0
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+

Z
dense_1/ConstConst*
valueB
*    *
dtype0*
_output_shapes
:

x
dense_1/bias
VariableV2*
shape:
*
shared_name *
dtype0*
	container *
_output_shapes
:

�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������
*
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������

g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*'
_output_shapes
:���������
*
T0
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
:���������

s
"dense_1/activity_regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
g
"dense_1/activity_regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
 dense_1/activity_regularizer/addAdd"dense_1/activity_regularizer/add/x dense_1/activity_regularizer/Sum*
T0*
_output_shapes
: 
m
dense_2/random_uniform/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *��!�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��!?*
dtype0
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
seed2�ߜ*
_output_shapes

:
*

seed
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes

:
*
T0
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes

:
*
T0
�
dense_2/kernel
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:

{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:

Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_3/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *�KF�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *�KF?*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seed2��*
_output_shapes

:*

seed*
T0*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:
�
dense_3/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:*
shape
:*
shared_name 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_3/TanhTanhdense_3/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_4/random_uniform/shapeConst*
valueB"   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *���*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
seed2�Ֆ*
_output_shapes

:+*

seed*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:+
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:+
�
dense_4/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:+*
shape
:+*
shared_name 
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+*
use_locking(
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:+
Z
dense_4/ConstConst*
dtype0*
_output_shapes
:+*
valueB+*    
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
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+*
use_locking(
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:+
�
dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������+*
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
Adam/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
s
Adam/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
	container *
_output_shapes
: 
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
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
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
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*
dtype0
o
Adam/beta_2
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: *
use_locking(
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
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
�
dense_4_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
q
dense_4_sample_weightsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
l
loss/dense_4_loss/subSubdense_4/Reludense_4_target*
T0*'
_output_shapes
:���������+
k
loss/dense_4_loss/SquareSquareloss/dense_4_loss/sub*
T0*'
_output_shapes
:���������+
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
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*
T0*#
_output_shapes
:���������
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
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*
Truncate( *

DstT0*#
_output_shapes
:���������*

SrcT0

a
loss/dense_4_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
loss/dense_4_loss/Mean_3Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
metrics/acc/ArgMaxArgMaxdense_4_targetmetrics/acc/ArgMax/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:���������*
T0	
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
}
training/Adam/gradients/ShapeConst*
dtype0*
_output_shapes
: *
_class
loc:@loss/add*
valueB 
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
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*
_class
loc:@loss/add*

index_type0
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
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Tshape0*
_output_shapes
:*
T0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0*
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
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1Shapeloss/dense_4_loss/truediv*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2Const*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB *
dtype0*
_output_shapes
: 
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
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB: *
dtype0
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
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
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Truncate( *

DstT0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������
�
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*
_output_shapes
:*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
valueB"      *
dtype0
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
Tshape0*
_output_shapes

:
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
out_type0*
_output_shapes
:*
T0
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*'
_output_shapes
:���������
*

Tmultiples0
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
out_type0*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*,
_class"
 loc:@loss/dense_4_loss/truediv*
valueB *
dtype0*
_output_shapes
: 
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������*
T0
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*#
_output_shapes
:���������
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
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
:*
	keep_dims( *

Tidx0
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeConst*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
valueB *
dtype0*
_output_shapes
: 
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1Shape dense_1/activity_regularizer/Abs*
_output_shapes
:*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
out_type0
�
Straining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeEtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*2
_output_shapes 
:���������:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������

�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Utraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:
�
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*'
_output_shapes
:���������

�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0
�
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*2
_output_shapes 
:���������:���������
�
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul
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
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*#
_output_shapes
:���������
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*'
_output_shapes
:���������
*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������

�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0
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
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
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
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*

index_type0*
_output_shapes
: 
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
N*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*#
_output_shapes
:���������*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Tshape0
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
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
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/CastCast@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1*

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Truncate( 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������*
T0
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
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
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
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_4_loss/Mean*

index_type0*
_output_shapes
: 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
N*
_output_shapes
:
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
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
Tshape0*0
_output_shapes
:������������������
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
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
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
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0
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
8training/Adam/gradients/loss/dense_4_loss/sub_grad/ShapeShapedense_4/Relu*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0
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
:training/Adam/gradients/loss/dense_4_loss/sub_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape*
T0*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*'
_output_shapes
:���������+
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
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*0
_output_shapes
:������������������*
T0
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
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul*
transpose_a(*
_output_shapes

:+
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*
T0*
_class
loc:@dense_3/Tanh*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(*
_output_shapes

:*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0*"
_class
loc:@dense_2/BiasAdd
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a( *'
_output_shapes
:���������
*
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(*
_output_shapes

:

�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������

�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������
*
T0
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:

�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a( *'
_output_shapes
:���������+*
transpose_b(
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a(*
_output_shapes

:+
*
transpose_b( 
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
training/Adam/add/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
_output_shapes
: *
T0
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
_output_shapes
: *
T0
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
training/Adam/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  �
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
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
h
training/Adam/zerosConst*
valueB+
*    *
dtype0*
_output_shapes

:+

�
training/Adam/Variable
VariableV2*
dtype0*
	container *
_output_shapes

:+
*
shape
:+
*
shared_name 
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+
*
use_locking(
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+

b
training/Adam/zeros_1Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_1
VariableV2*
dtype0*
	container *
_output_shapes
:
*
shape:
*
shared_name 
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:
*
T0
j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_2
VariableV2*
dtype0*
	container *
_output_shapes

:
*
shape
:
*
shared_name 
�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
_output_shapes

:
*
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_3
VariableV2*
	container *
_output_shapes
:*
shape:*
shared_name *
dtype0
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_3
j
training/Adam/zeros_4Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_4
VariableV2*
dtype0*
	container *
_output_shapes

:*
shape
:*
shared_name 
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
_output_shapes

:*
T0*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_5
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
j
training/Adam/zeros_6Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
_output_shapes

:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:+
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
dtype0*
	container *
_output_shapes
:+*
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
j
training/Adam/zeros_8Const*
dtype0*
_output_shapes

:+
*
valueB+
*    
�
training/Adam/Variable_8
VariableV2*
dtype0*
	container *
_output_shapes

:+
*
shape
:+
*
shared_name 
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
validate_shape(*
_output_shapes

:+
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+

b
training/Adam/zeros_9Const*
_output_shapes
:
*
valueB
*    *
dtype0
�
training/Adam/Variable_9
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:
*
shape:

�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:

�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_9
k
training/Adam/zeros_10Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_10
VariableV2*
dtype0*
	container *
_output_shapes

:
*
shape
:
*
shared_name 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:

c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
k
training/Adam/zeros_12Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_12
VariableV2*
dtype0*
	container *
_output_shapes

:*
shape
:*
shared_name 
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
_output_shapes

:*
T0*,
_class"
 loc:@training/Adam/Variable_12
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_13
k
training/Adam/zeros_14Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_14
VariableV2*
dtype0*
	container *
_output_shapes

:+*
shape
:+*
shared_name 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:+
c
training/Adam/zeros_15Const*
dtype0*
_output_shapes
:+*
valueB+*    
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
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:
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
training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*
_output_shapes
:*
T0*

index_type0
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
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_17
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
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_18
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
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
_output_shapes
:*
T0*

index_type0
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
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:
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
training/Adam/zeros_20/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_20
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
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
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_20
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
training/Adam/zeros_21Fill&training/Adam/zeros_21/shape_as_tensortraining/Adam/zeros_21/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_21
VariableV2*
dtype0*
	container *
_output_shapes
:*
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
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*
_output_shapes
:
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
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23*
validate_shape(*
_output_shapes
:
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

:+

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

:+

m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+

t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes

:+
*
T0
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
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:+
*
T0
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes

:+
*
T0
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

:+
*
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+

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
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*
_output_shapes

:+

�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
_output_shapes

:+
*
T0
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
_output_shapes

:+
*
T0
Z
training/Adam/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
p
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
_output_shapes

:+
*
T0
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

:+
*
T0
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
_output_shapes

:+
*
T0
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+

�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+

�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+
*
use_locking(*
T0
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:

Z
training/Adam/sub_5/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:

p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes
:
*
T0
Z
training/Adam/sub_6/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:

i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:
*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:

Z
training/Adam/Const_4Const*
dtype0*
_output_shapes
: *
valueB
 *    
Z
training/Adam/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
_output_shapes
:
*
T0
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:

`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:

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
:

r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:

k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:

�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:

�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:

�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:

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

:
*
T0
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:

v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:

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
training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:
*
T0
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:

l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:

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

:

�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
_output_shapes

:
*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
_output_shapes

:
*
T0
Z
training/Adam/add_9/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*
_output_shapes

:

v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:

r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:

�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:
*
use_locking(
�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:
*
use_locking(
�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:

q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:
[
training/Adam/sub_11/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:
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
:
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:*
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes
:*
T0
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  �*
dtype0*
_output_shapes
: 
�
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
:
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
:*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:
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
:
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes
:*
T0
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:
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

:
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:
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
training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:
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
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
_output_shapes

:*
T0
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:
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

:*
T0
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
_output_shapes

:*
T0
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes
:*
T0
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
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes
:*
T0
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:
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
:
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
:*
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes
:
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
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
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes
:*
T0
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
:
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:
[
training/Adam/add_18/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
:*
T0
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:+
[
training/Adam/sub_20/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:+*
T0
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes

:+*
T0
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:+
[
training/Adam/sub_21/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:+
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes

:+*
T0
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:+
[
training/Adam/Const_14Const*
_output_shapes
: *
valueB
 *    *
dtype0
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

:+
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes

:+*
T0
[
training/Adam/add_21/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes

:+*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:+
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

:+*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
_output_shapes

:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
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
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes
:+*
T0
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
_output_shapes
:+*
T0
[
training/Adam/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:+*
T0
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes
:+*
T0
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:+
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
training/Adam/Const_17Const*
_output_shapes
: *
valueB
 *  �*
dtype0
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
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes
:+*
T0
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
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+*
use_locking(
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
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_3/bias
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*
_output_shapes
: *!
_class
loc:@dense_4/kernel*
dtype0
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
IsVariableInitialized_9IsVariableInitializedAdam/lr*
_output_shapes
: *
_class
loc:@Adam/lr*
dtype0
�
IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_2
�
IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0
�
IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*
_output_shapes
: *)
_class
loc:@training/Adam/Variable*
dtype0
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
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_8
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
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_11
�
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_12*
dtype0
�
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_14*
dtype0
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
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_17
�
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_18
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
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_21*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_21
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
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08"� 
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
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08'�       ���	��ÁB=�A*

val_loss���?Gx=�       �	E�ÁB=�A*

val_acc])�=O1%       �K"	��ÁB=�A*

loss��@�,�u       ���	��ÁB=�A*


accjyv=��D�       ��2	ŋ́B=�A*

val_lossߙ�?t+       `/�#	�́B=�A*

val_acc+l=إ�       ��-	X�́B=�A*

loss�L�?�       ��(	��́B=�A*


accW˄=���       ��2	��ׁB=�A*

val_loss�?�6�       `/�#	��ׁB=�A*

val_acc�"�=��6       ��-	A�ׁB=�A*

loss!�?�E'^       ��(	��ׁB=�A*


acc�|=�!�       ��2	��B=�A*

val_loss*[�?�>o       `/�#	��B=�A*

val_acc�f�=a��n       ��-	L�B=�A*

loss��?d���       ��(	��B=�A*


acc˄�=�52�       ��2	 �B=�A*

val_loss`��?iR��       `/�#	��B=�A*

val_acc�f�=�O�       ��-	q�B=�A*

lossa6�?���X       ��(	 �B=�A*


acc��=[�W       ��2	i��B=�A*

val_loss�e?����       `/�#	Dj��B=�A*

val_acc�=�=�j�       ��-	�j��B=�A*

loss�?��!       ��(	k��B=�A*


acc盏=��|       ��2	z4��B=�A*

val_loss,]u?���       `/�#	�6��B=�A*

val_accO�Y=��       ��-	a7��B=�A*

loss�#�?�,fd       ��(	�7��B=�A*


acc`�b=�2�3       ��2	M�B=�A*

val_lossP o?��X7       `/�#	4N�B=�A*

val_acc��==���9       ��-	�N�B=�A*

loss9=z?3�&�       ��(	�N�B=�A*


acc��4=�Ґ�       ��2	�j�B=�A*

val_lossi�i?ZDXn       `/�#	�k�B=�A*

val_acc��/=iR��       ��-	rl�B=�A*

loss�ct?{85�       ��(	�l�B=�A*


acc1=@��       ��2	5��B=�A	*

val_lossaCe?���       `/�#	���B=�A	*

val_acc1k!=���       ��-	��B=�A	*

loss`o?�0Y�       ��(	���B=�A	*


acc�0=��       ��2	�!�B=�A
*

val_loss޽`?����       `/�#	�!�B=�A
*

val_acc���<�!Q       ��-	i!�B=�A
*

lossX�j?s�s}       ��(	�!�B=�A
*


acc�[=���!       ��2	_{)�B=�A*

val_loss�\?6h_�       `/�#	k|)�B=�A*

val_acc1k!=�\�X       ��-	�|)�B=�A*

lossj�e?�^4       ��(	9})�B=�A*


acc�	=_4�       ��2	yD5�B=�A*

val_loss�+Y?��2�       `/�#	�F5�B=�A*

val_acc��9=UE�K       ��-	9G5�B=�A*

loss��a?/��*       ��(	�G5�B=�A*


acc)�D=���m       ��2	A@�B=�A*

val_loss25V?x�V       `/�#	�@�B=�A*

val_acc�M=y�,       ��-	@�B=�A*

lossLm^?Q��       ��(	�@�B=�A*


acc�FU=x�b       ��2	�M�B=�A*

val_lossΔS?�ۯ       `/�#	��M�B=�A*

val_acc��K=�;       ��-	R�M�B=�A*

lossؓ[?�!y;       ��(	�M�B=�A*


acc�R]=�	��       ��2	[�V�B=�A*

val_loss@zQ?8$}       `/�#	 �V�B=�A*

val_acc&�Q=ڣ�]       ��-	�V�B=�A*

lossDY?�+^       ��(	��V�B=�A*


accqT^=�'�       ��2	�;`�B=�A*

val_lossy�O?���       `/�#	A`�B=�A*

val_acc&�Q=�`�       ��-	D`�B=�A*

loss�W?h��       ��(	xE`�B=�A*


accqT^=P�8^       ��2	�&i�B=�A*

val_lossm�N?=�=�       `/�#	�'i�B=�A*

val_acc&�Q=¿Q�       ��-	 (i�B=�A*

loss�_U?��^0       ��(	�(i�B=�A*


accqT^=����       ��2	0�r�B=�A*

val_loss(*M?4��-       `/�#	��r�B=�A*

val_acc&�Q=S��       ��-	��r�B=�A*

loss��S?��       ��(	v�r�B=�A*


accqT^=k�ڑ       ��2	/�{�B=�A*

val_loss�KL?WP*       `/�#	e�{�B=�A*

val_acc&�Q=6w|�       ��-	��{�B=�A*

loss/�R?��/       ��(	� |�B=�A*


accqT^=ۯC�       ��2	�g��B=�A*

val_loss�zK?ا�}       `/�#	�i��B=�A*

val_acc;�U=/�Kt       ��-	�k��B=�A*

loss֥Q?wF�       ��(	�l��B=�A*


accqT^=؄��       ��2	�9��B=�A*

val_loss�`J?rX��       `/�#	�<��B=�A*

val_acc;�U=J�       ��-	6?��B=�A*

loss<yP?R���       ��(	A��B=�A*


acc6�^=H��+       ��2	8ܟ�B=�A*

val_loss��I?\��[       `/�#	�ݟ�B=�A*

val_acc��O=i�zc       ��-	#ޟ�B=�A*

loss�}O?�E��       ��(	�ޟ�B=�A*


acc6�^=I� �       ��2	����B=�A*

val_loss�H?�=�T       `/�#	����B=�A*

val_acc��S=#I��       ��-	���B=�A*

loss�N?[{v�       ��(	f���B=�A*


acc6�^=J�q�       ��2	����B=�A*

val_loss�H?�߯3       `/�#	�Ĵ�B=�A*

val_acc��S=/��B       ��-	�ƴ�B=�A*

losss�M?Pt�N       ��(	�ȴ�B=�A*


acc6�^=�\       ��2	��ÂB=�A*

val_loss��G?l�m       `/�#	M�ÂB=�A*

val_acc��O=__�       ��-	@�ÂB=�A*

loss�M?�/       ��(	��ÂB=�A*


acc�U_=�q[       ��2	�ςB=�A*

val_loss�aG?�'x�       `/�#	�ςB=�A*

val_acc��[=5Ud       ��-	�ςB=�A*

loss[;L?˜c       ��(	�ςB=�A*


acc�U_=i�o       ��2	w�܂B=�A*

val_loss'�E?��&       `/�#	�܂B=�A*

val_acc;�U=_��       ��-	��܂B=�A*

loss�>K?���       ��(	U�܂B=�A*


accqT^=^5�       ��2	D��B=�A*

val_loss��E?�}r       `/�#	f��B=�A*

val_acc�E=�_l       ��-	Τ�B=�A*

loss�YJ?Uս       ��(	"��B=�A*


acc6�^=�3n       ��2	B���B=�A*

val_loss�D?'��       `/�#	|���B=�A*

val_acc��[=0�B�       ��-	1���B=�A*

loss�I?����       ��(	����B=�A*


acc6�^=_}1       ��2	�`�B=�A*

val_lossŗC?�or�       `/�#	b�B=�A*

val_acc_�C=T�k�       ��-	�b�B=�A*

loss�}H?��h�       ��(	�b�B=�A*


accHNZ=Lqx�       ��2	�`�B=�A*

val_loss��B?~��       `/�#	�a�B=�A*

val_acc_�C=�ܬ       ��-	Wb�B=�A*

loss�qG?�&:       ��(	�b�B=�A*


acc�2H=Pֶ�       ��2	���B=�A *

val_loss�(B?ۚ�       `/�#	���B=�A *

val_acc�4	=�`�       ��-	T��B=�A *

lossyF?���       ��(	���B=�A *


acc"�7=�%�       ��2	��B=�A!*

val_loss�LA?�RS�       `/�#	��B=�A!*

val_acc�)�<(I��       ��-	�!�B=�A!*

lossDxE?�7ק       ��(	W"�B=�A!*


acc-�%=$�ݰ       ��2	��'�B=�A"*

val_loss��@?�:��       `/�#	�'�B=�A"*

val_acc��/=?��       ��-	я'�B=�A"*

loss��D?\�IH       ��(	e�'�B=�A"*


acc?w!= �3R       ��2	��/�B=�A#*

val_loss�@?�	��       `/�#	-�/�B=�A#*

val_accY=Ǥ�s       ��-	�/�B=�A#*

loss��C?��l       ��(	}�/�B=�A#*


acc�g=ݙ�n       ��2	I�7�B=�A$*

val_loss|e??�t�       `/�#	˾7�B=�A$*

val_acc6�;=��D�       ��-	׿7�B=�A$*

loss�_C?4\2       ��(	��7�B=�A$*


acc�[=,�?       ��2	��A�B=�A%*

val_loss�>?���~       `/�#	��A�B=�A%*

val_acc�4	=��1�       ��-	E�A�B=�A%*

lossZlB?|�Z�       ��(	��A�B=�A%*


acc��=�j�       ��2	�DN�B=�A&*

val_loss�I>?r���       `/�#	JFN�B=�A&*

val_accx��<��R       ��-	�FN�B=�A&*

loss�A?uZk<       ��(	�GN�B=�A&*


acc�a=[*u       ��2	>V�B=�A'*

val_loss��=?p��Y       `/�#	�?V�B=�A'*

val_acc�x'=8�       ��-	1@V�B=�A'*

loss��@?�L�v       ��(	�@V�B=�A'*


acce�=���       ��2	��]�B=�A(*

val_loss�=?H�=       `/�#	��]�B=�A(*

val_acc�O=��4�       ��-	�]�B=�A(*

loss�@?�A�j       ��(	m�]�B=�A(*


acc�r="x��       ��2	�l�B=�A)*

val_loss�6=?6�B       `/�#	�l�B=�A)*

val_acc,0=��.       ��-	�l�B=�A)*

loss��??��Qh       ��(	m!l�B=�A)*


acc�(=����       ��2	#y�B=�A**

val_lossa�<?�L�o       `/�#	�y�B=�A**

val_accJ�?=�D�       ��-	�y�B=�A**

loss�v??�ଝ       ��(	�y�B=�A**


acca�==�ځg       ��2	����B=�A+*

val_lossa<?����       `/�#	����B=�A+*

val_acc�)v=Zň       ��-	+���B=�A+*

loss+�>?y��E       ��(	����B=�A+*


acc�Z=�&�       ��2	�p��B=�A,*

val_loss�<?��Ω       `/�#	�t��B=�A,*

val_acc�+�=����       ��-	�u��B=�A,*

loss϶>?I�bj       ��(	mv��B=�A,*


acc��l=�X�       ��2	���B=�A-*

val_loss_�;?0�       `/�#	\��B=�A-*

val_acc%I�=i�?y       ��-	\��B=�A-*

lossO6>?�s/'       ��(	 ��B=�A-*


acc��=r�/       ��2	hW��B=�A.*

val_loss�g;?N��       `/�#	�Z��B=�A.*

val_acc<~=2��E       ��-	`[��B=�A.*

lossq�=?��DU       ��(	�[��B=�A.*


accCȂ=%���       ��2	R���B=�A/*

val_loss%0;?3�6       `/�#	'���B=�A/*

val_accM[�=�&�       ��-	���B=�A/*

loss_=?��4R       ��(	����B=�A/*


acc盏=�k       ��2	-�B=�A0*

val_loss��:?�%��       `/�#	��B=�A0*

val_accr�=^H       ��-	o�B=�A0*

loss��<?�*�j       ��(	��B=�A0*


acct'�=Z#M�       ��2	e�փB=�A1*

val_loss�:?���       `/�#	�׃B=�A1*

val_accbd�=�T       ��-	�׃B=�A1*

losso�<?"	�       ��(	�׃B=�A1*


acc٬�=���       ��2	ף�B=�A2*

val_loss�<:?��       `/�#	/��B=�A2*

val_acc�]�=�S       ��-	��B=�A2*

loss�W<?���       ��(	���B=�A2*


accc��=[�B       ��2	F��B=�A3*

val_loss�:?'�j       `/�#	>��B=�A3*

val_acc~T�=��{�       ��-	���B=�A3*

loss<?����       ��(	��B=�A3*


acc=2�=E�!M       ��2	����B=�A4*

val_loss�(:?�i       `/�#	2��B=�A4*

val_acc)��=�mBv       ��-	!��B=�A4*

lossk<?/�       ��(	"��B=�A4*


acc,�=6�E       ��2	���B=�A5*

val_loss�9?�w�       `/�#	&��B=�A5*

val_acc"��=菹       ��-	���B=�A5*

loss]�;?Y�`       ��(	���B=�A5*


acc*t�=Qb�C       ��2	��B=�A6*

val_loss�}9?��       `/�#	Z��B=�A6*

val_acco��=���       ��-	3��B=�A6*

loss�V;?6�*       ��(	���B=�A6*


acc�r�=��F       ��2	���B=�A7*

val_loss�q9?v��       `/�#	C��B=�A7*

val_accr�=���        ��-	���B=�A7*

loss��:?{�A�       ��(	���B=�A7*


accU��=?���       ��2	en#�B=�A8*

val_lossT�9?�       `/�#	p#�B=�A8*

val_acc1k�=��K~       ��-	�p#�B=�A8*

lossL�:?��       ��(	�p#�B=�A8*


acch}�=��6(       ��2	��.�B=�A9*

val_loss�(9?���       `/�#	��.�B=�A9*

val_acc"��=W^'       ��-	q�.�B=�A9*

loss��:?��9       ��(	��.�B=�A9*


acc���=g]��       ��2	/�7�B=�A:*

val_loss�.9?�t��       `/�#	*�7�B=�A:*

val_acc䁫=�
'       ��-	��7�B=�A:*

loss%C:?��+C       ��(	�7�B=�A:*


acc}��=׺�N       ��2	�E�B=�A;*

val_loss0�8?�5v       `/�#	Q�E�B=�A;*

val_accg��=))Ø       ��-	+�E�B=�A;*

loss2:?9}?�       ��(	ʿE�B=�A;*


acc/C�=Ihs�       ��2	��N�B=�A<*

val_lossv�8?]W�3       `/�#	��N�B=�A<*

val_acc���=����       ��-	#�N�B=�A<*

loss��9?�r�       ��(	��N�B=�A<*


acc~Ū=� ��       ��2	LTU�B=�A=*

val_loss4�8?{{�       `/�#	mUU�B=�A=*

val_acc���=���V       ��-	�UU�B=�A=*

lossR�9?��m�       ��(	XVU�B=�A=*


accW�=��<�       ��2	�a�B=�A>*

val_lossT8?c�       `/�#	̛a�B=�A>*

val_acc�=���X       ��-	��a�B=�A>*

loss�c9?��cG       ��(	�a�B=�A>*


acc���=�xx�       ��2	��k�B=�A?*

val_loss��7?K�K       `/�#	s�k�B=�A?*

val_acc���=ON�       ��-	0�k�B=�A?*

lossL9?��K�       ��(	��k�B=�A?*


accmL�=�a��       ��2	]�u�B=�A@*

val_lossk�7?����       `/�#	��u�B=�A@*

val_acc��=$n�Y       ��-	n�u�B=�A@*

loss	9?:���       ��(	��u�B=�A@*


accE��=~��]       ��2	耄B=�AA*

val_lossa�7?�Jӭ       `/�#	)逄B=�AA*

val_accg��=˜_{       ��-	�逄B=�AA*

loss��8?�ج�       ��(	ꀄB=�AA*


acc4�=x�;=       ��2	�0��B=�AB*

val_lossIK7?k:��       `/�#	
2��B=�AB*

val_accR��=�       ��-	�2��B=�AB*

loss�8?	:@       ��(	�2��B=�AB*


acc��=���       ��2	c���B=�AC*

val_loss�7?��Qt       `/�#	���B=�AC*

val_acck��=O���       ��-	蹕�B=�AC*

losszZ8?k�p�       ��(	k���B=�AC*


acc4�==��B       ��2	x��B=�AD*

val_loss�"7?k��       `/�#	A��B=�AD*

val_accܚ�=�K�%       ��-	,��B=�AD*

lossI/8?�@j       ��(	���B=�AD*


acc��=�Ͻn       ��2	a���B=�AE*

val_loss�/7?��IT       `/�#	����B=�AE*

val_acc���=�^        ��-	����B=�AE*

loss��7?�Ρo       ��(	����B=�AE*


acc'h�=��       ��2	�ö�B=�AF*

val_lossP�6?��T       `/�#	:Ƕ�B=�AF*

val_acc���=�?��       ��-	iɶ�B=�AF*

loss`�7?5���       ��(	�ʶ�B=�AF*


acc+��=��8�       ��2	� ��B=�AG*

val_lossM�6?����       `/�#	�!��B=�AG*

val_acc���=�'M       ��-	>"��B=�AG*

loss�|7?�1�"       ��(	�"��B=�AG*


acc,7�=�R�       ��2	�w̄B=�AH*

val_lossz6?��       `/�#	�{̄B=�AH*

val_acc���=��m       ��-	W}̄B=�AH*

loss�N7?��E       ��(	�~̄B=�AH*


accv�=��       ��2	�iԄB=�AI*

val_lossd�6?R��j       `/�#	?kԄB=�AI*

val_acc&��=) �A       ��-	�kԄB=�AI*

loss�7?�5eC       ��(	DlԄB=�AI*


accF	�=ɷ`�       ��2	kAބB=�AJ*

val_loss#�6?��6       `/�#	dBބB=�AJ*

val_acc�=>|�8�       ��-	�BބB=�AJ*

lossU7?z�J       ��(	JCބB=�AJ*


acc���={H�_       ��2	.��B=�AK*

val_loss�]6?�`��       `/�#	K��B=�AK*

val_acc9R>�k)r       ��-	ͭ�B=�AK*

loss��6?��"�       ��(	2��B=�AK*


acc�5�=���o       ��2	`��B=�AL*

val_loss��5?IT�#       `/�#	���B=�AL*

val_acc��>a�*l       ��-	u��B=�AL*

lossf�6?���0       ��(	S��B=�AL*


accv->v�)�       ��2	C���B=�AM*

val_loss6?�!1       `/�#	6���B=�AM*

val_accI  >/zˠ       ��-	����B=�AM*

loss�K6?��hV       ��(	���B=�AM*


accq�>`Ꙕ       ��2	cD�B=�AN*

val_loss��5?㷲       `/�#	1F�B=�AN*

val_acc>�0>�E�       ��-	�F�B=�AN*

loss�o6?�R�       ��(	{G�B=�AN*


acc�>��.       ��2	Wz�B=�AO*

val_loss�#6?��O�       `/�#	>{�B=�AO*

val_acc%I>���T       ��-	�{�B=�AO*

loss�)6?Iw��       ��(	�{�B=�AO*


acc�a(>���       ��2	���B=�AP*

val_loss�'5?m�e       `/�#	_��B=�AP*

val_acc��&>�q�       ��-	��B=�AP*

lossX6?�!T�       ��(	���B=�AP*


accz55>�|�t       ��2	_ �B=�AQ*

val_loss|55?�PB�       `/�#	� �B=�AQ*

val_acc�2A>Q��       ��-	� �B=�AQ*

loss<�5?q��;       ��(	4 �B=�AQ*


accm�C>�@��       ��2	�..�B=�AR*

val_loss��4?�I�       `/�#	�/.�B=�AR*

val_acck�R>�B�G       ��-	,0.�B=�AR*

loss�{5?"��       ��(	�0.�B=�AR*


acc�RH>�;�7       ��2	2v6�B=�AS*

val_loss5?�v�       `/�#	�w6�B=�AS*

val_acc GJ>qB�@       ��-	yx6�B=�AS*

lossV�5?�b�       ��(	y6�B=�AS*


accS> N       ��2	i�A�B=�AT*

val_lossʸ4?X��j       `/�#	�A�B=�AT*

val_acc>�0>�͉       ��-	S�A�B=�AT*

loss�F5?PkK�       ��(	��A�B=�AT*


acc�FU>47�       ��2	��H�B=�AU*

val_loss��4?Rw\t       `/�#	��H�B=�AU*

val_acc�m[>�9��       ��-	"�H�B=�AU*

lossm5?Dk�)       ��(	~�H�B=�AU*


accn�W>��&Y       ��2	6T�B=�AV*

val_loss��4?Ƙ�       `/�#	�T�B=�AV*

val_acc_�C>	�<U       ��-	1T�B=�AV*

lossA�4? �;/       ��(	�T�B=�AV*


acc�a>�l��       ��2	�"[�B=�AW*

val_loss�4?}��       `/�#	%[�B=�AW*

val_acc{�<>�%p�       ��-	&[�B=�AW*

loss@�4?W�       ��(	�&[�B=�AW*


acc5�o>iA(�       ��2	quh�B=�AX*

val_loss��3?�	�g       `/�#	�vh�B=�AX*

val_acc}7|>��&       ��-	Swh�B=�AX*

loss��4?�ˇ       ��(	�wh�B=�AX*


acc;�h>�^��       ��2	tq�B=�AY*

val_loss��4?�O�       `/�#	�uq�B=�AY*

val_acc85{>���       ��-	�vq�B=�AY*

loss�4?ELÏ       ��(	�wq�B=�AY*


acc��l>Ҧܪ       ��2	�7}�B=�AZ*

val_loss�3?�^N        `/�#	�:}�B=�AZ*

val_acc�+�>��@*       ��-	C=}�B=�AZ*

loss	O4?�'Q�       ��(	>}�B=�AZ*


accSr>�V       ��2	�/��B=�A[*

val_loss	�3?����       `/�#	�0��B=�A[*

val_acc�>�6/�       ��-	j1��B=�A[*

loss��3?q"q@       ��(	�1��B=�A[*


accs�u>#��]       ��2	x��B=�A\*

val_loss3D4?�ɱ       `/�#	���B=�A\*

val_acc3�`>/�x       ��-	g��B=�A\*

losso�3?�_��       ��(	���B=�A\*


acc͹v>��[]       ��2	h���B=�A]*

val_loss�3?���       `/�#	����B=�A]*

val_accc�>��       ��-	���B=�A]*

loss��3?����       ��(	c���B=�A]*


accD�x><	��       ��2	�6��B=�A^*

val_loss��3?_��       `/�#	G9��B=�A^*

val_acc@p>�7�       ��-	�9��B=�A^*

lossĔ3?e��i       ��(	�:��B=�A^*


acc˄�>\I�C       ��2	����B=�A_*

val_loss��3?-9�       `/�#	����B=�A_*

val_acc��P>�l8       ��-		���B=�A_*

lossF�3?3�m�       ��(	����B=�A_*


acc��>/��b       ��2	����B=�A`*

val_loss�Y4?�.��       `/�#	����B=�A`*

val_acc�f>t}�        ��-	J���B=�A`*

loss�3?9�CC       ��(	����B=�A`*


acc<�|>|y       ��2	[�ÅB=�Aa*

val_lossݫ3?��y       `/�#	��ÅB=�Aa*

val_accd�]>�`       ��-	�ÅB=�Aa*

loss��3?�C��       ��(	8�ÅB=�Aa*


acc�>�@�x       ��2	�΅B=�Ab*

val_losss�3?�'��       `/�#	�΅B=�Ab*

val_acc�o>W�[       ��-	S΅B=�Ab*

loss�k3?J�c@       ��(	�΅B=�Ab*


accwb|>��s5       ��2	�CՅB=�Ac*

val_loss��3?ֱn�       `/�#	lDՅB=�Ac*

val_acc�z>��~�       ��-	�DՅB=�Ac*

loss? 3?>Y �       ��(	$EՅB=�Ac*


acc��>Ͼ�       ��2	�r��B=�Ad*

val_loss��2?d�p�       `/�#	.t��B=�Ad*

val_acc�-�>���       ��-	�t��B=�Ad*

loss�2?}c��       ��(	uu��B=�Ad*


acc��>^K#n       ��2	q��B=�Ae*

val_lossE<3?�
'K       `/�#	:��B=�Ae*

val_acc���>BW�       ��-	��B=�Ae*

loss��2?d�]9       ��(	\��B=�Ae*


accj��>���       ��2	����B=�Af*

val_lossGC3?�DT�       `/�#	���B=�Af*

val_acc�i�>L�I+       ��-	`���B=�Af*

loss�2?��-       ��(	����B=�Af*


accv-�>�ԥ       ��2	����B=�Ag*

val_loss[�2?;��       `/�#	^���B=�Ag*

val_acc�+�>ψ�       ��-	/���B=�Ag*

lossf�2?�p	G       ��(	����B=�Ag*


accL'�>�a�\       ��2	v�B=�Ah*

val_loss��2?_G��       `/�#	��B=�Ah*

val_accyn�>x�˚       ��-	e�B=�Ah*

loss�2?�Ȱ       ��(	�B=�Ah*


acc��>��\D       ��2	��B=�Ai*

val_loss �3?���       `/�#	|�B=�Ai*

val_acc�0y>�r!       ��-	��B=�Ai*

loss)Z2?��[}       ��(	M�B=�Ai*


accɼ�>]c3�       ��2	;��B=�Aj*

val_loss>�2?�J!�       `/�#	&��B=�Aj*

val_accO��>yps3       ��-	���B=�Aj*

lossR�2?�}�v       ��(	���B=�Aj*


accz:�>`���       ��2	#��B=�Ak*

val_lossb�2?ϔ(k       `/�#	���B=�Ak*

val_acc|�>^���       ��-	A��B=�Ak*

loss�2?:�       ��(	���B=�Ak*


acc?��>��       ��2	�0*�B=�Al*

val_loss 02?�B�[       `/�#	�2*�B=�Al*

val_acc4l�>kl��       ��-	Y4*�B=�Al*

loss�"2?�,&�       ��(	�5*�B=�Al*


acc���>�u�d       ��2	)�1�B=�Am*

val_loss��2?ٛGX       `/�#	:�1�B=�Am*

val_acc�'u>a+y&       ��-	ɕ1�B=�Am*

loss��1?���       ��(	5�1�B=�Am*


accL�>eJ��       ��2	��=�B=�An*

val_loss�1?$��>       `/�#	�=�B=�An*

val_accc�>k�*?       ��-	t�=�B=�An*

loss��1?^9�%       ��(	�=�B=�An*


accwr�>tf��       ��2	ȖE�B=�Ao*

val_loss�E2?}`��       `/�#	��E�B=�Ao*

val_acc|�> w8       ��-	��E�B=�Ao*

loss��1?���\       ��(	��E�B=�Ao*


accNl�>��Ҽ       ��2	Q�B=�Ap*

val_loss��1?ٰ$�       `/�#	�Q�B=�Ap*

val_accٵ�>��J       ��-	@Q�B=�Ap*

loss.�1?���       ��(	{Q�B=�Ap*


acc���>�O       ��2	D�W�B=�Aq*

val_loss��1?��6       `/�#	/�W�B=�Aq*

val_acc��>K�	�       ��-	��W�B=�Aq*

loss�o1?��*       ��(	 �W�B=�Aq*


acc���>TMj�       ��2	\c�B=�Ar*

val_loss}1??��       `/�#	�c�B=�Ar*

val_acc���>�^3       ��-	lc�B=�Ar*

loss�z1?��s�       ��(	�	c�B=�Ar*


acc���>��       ��2	.�i�B=�As*

val_lossI�1?�㛎       `/�#	6�i�B=�As*

val_acc�4�>����       ��-	��i�B=�As*

loss+H1?��q       ��(	�i�B=�As*


acc%�>]v$:       ��2	��u�B=�At*

val_lossw1?(��       `/�#	W�u�B=�At*

val_acc�y�>�d       ��-	�u�B=�At*

loss'1?��	?       ��(	��u�B=�At*


acc)5�>".�       ��2	� ��B=�Au*

val_loss�V1?3��
       `/�#	y��B=�Au*

val_acc�=�>��       ��-	G��B=�Au*

loss�1?�+��       ��(	���B=�Au*


acc<�>JL5       ��2	�5��B=�Av*

val_lossC"1?/�74       `/�#	78��B=�Av*

val_acc���>r���       ��-	9��B=�Av*

lossU�0?s��       ��(	�9��B=�Av*


acc��>�ib       ��2	�є�B=�Aw*

val_loss��1?sۛ       `/�#	�Ҕ�B=�Aw*

val_acc�-�>I���       ��-	cӔ�B=�Aw*

lossh1?|�       ��(	�Ӕ�B=�Aw*


acc�Ԋ>6���       ��2	�@��B=�Ax*

val_lossv�1?��?�       `/�#	�A��B=�Ax*

val_acc;��>�b8�       ��-	kB��B=�Ax*

lossn1?��	�       ��(	�B��B=�Ax*


accQ��>nMW       ��2	l���B=�Ay*

val_lossn[1?��       `/�#	K���B=�Ay*

val_acc|�>���       ��-	����B=�Ay*

loss C1?���       ��(	���B=�Ay*


acc�#�>P�       ��2	,���B=�Az*

val_lossq�1?FEn       `/�#	Ԝ��B=�Az*

val_acc9R�>�T�u       ��-	g���B=�Az*

loss��0?\��       ��(	ԝ��B=�Az*


acc��>�"       ��2	�,��B=�A{*

val_loss�1?�+1       `/�#	�.��B=�A{*

val_accc��>~ը,       ��-	�/��B=�A{*

loss,�0?�|�       ��(	40��B=�A{*


acc��>�#F�       ��2	�ĆB=�A|*

val_loss�0?��>l       `/�#	wĆB=�A|*

val_accUB�>�HO       ��-	LĆB=�A|*

lossNe0?��U       ��(	�ĆB=�A|*


accb�>�?��       ��2	�GφB=�A}*

val_loss�0?k��       `/�#	JφB=�A}*

val_acc�i�>�@��       ��-	�JφB=�A}*

lossy"0?�ō       ��(	�KφB=�A}*


acc�ݐ>x�Z�       ��2	|�ՆB=�A~*

val_loss��0?9^�       `/�#	��ՆB=�A~*

val_acc���>fI~�       ��-	�ՆB=�A~*

lossl0?�
/       ��(	p�ՆB=�A~*


acc��>(�0       ��2	�݆B=�A*

val_loss�[1?��5       `/�#	݆B=�A*

val_acc���>e¸�       ��-	�݆B=�A*

loss��0?�9�       ��(	݆B=�A*


accse�>J���       QKD	!�B=�A�*

val_loss��0?��<       ��2	��B=�A�*

val_acc:��>Z�G       �	�B=�A�*

loss�0?�q�q       ��-	h�B=�A�*


acc�>s��       QKD	!��B=�A�*

val_loss~v0?��,       ��2	���B=�A�*

val_accj��>�=       �	-��B=�A�*

loss@�/?L�        ��-	-��B=�A�*


accu��>����       QKD	���B=�A�*

val_lossmX0?�3�=       ��2	��B=�A�*

val_acc�p�>��>*       �	���B=�A�*

loss��/?E���       ��-	-	��B=�A�*


acc6�>����       QKD	���B=�A�*

val_loss50?�)�       ��2	���B=�A�*

val_acc�6�>�z5�       �	���B=�A�*

loss_�/?H54       ��-	��B=�A�*


acc�V�>*���       QKD	Ψ�B=�A�*

val_lossh,1?;�PD       ��2	���B=�A�*

val_acc��>��ɼ       �	2��B=�A�*

loss~�/?`�`g       ��-	���B=�A�*


acc���>c8�       QKD	�l�B=�A�*

val_loss^�0?�d�       ��2	eq�B=�A�*

val_accB�>�D%=       �	�s�B=�A�*

loss�	0?�͇�       ��-	>u�B=�A�*


accM�>)�       QKD	��#�B=�A�*

val_loss�!0?��^�       ��2	p�#�B=�A�*

val_acc��>ˑ��       �	J�#�B=�A�*

loss��/? �1u       ��-	�#�B=�A�*


acc�>Nlvt       QKD	��.�B=�A�*

val_loss0?�ܸ       ��2	��.�B=�A�*

val_acc
��>L'E       �	��.�B=�A�*

loss�/?.��1       ��-	��.�B=�A�*


acc�Y�>̇dz       QKD	�6�B=�A�*

val_loss��0?nɶ�       ��2	9�6�B=�A�*

val_acc���>�sZ�       �	��6�B=�A�*

lossp�/?;Yz�       ��-	�6�B=�A�*


acc�9�>��9       QKD	�.A�B=�A�*

val_lossY�0?��G�       ��2	b2A�B=�A�*

val_acc3��>�l,l       �	Y3A�B=�A�*

losse�/?��       ��-	4A�B=�A�*


accj�>ݮ�       QKD	txJ�B=�A�*

val_lossq0?F��J       ��2	lyJ�B=�A�*

val_accȑ>����       �	�yJ�B=�A�*

loss:�/?�L}%       ��-	FzJ�B=�A�*


acc�9�>R��        QKD	�T�B=�A�*

val_loss��/?�%��       ��2	��T�B=�A�*

val_accj��>,"Q       �	��T�B=�A�*

loss�M/?>�h�       ��-	0�T�B=�A�*


acc���>t�       QKD	[�^�B=�A�*

val_loss��/?W�ۑ       ��2	��^�B=�A�*

val_acc���>�6�'       �	��^�B=�A�*

lossM!/?�\#       ��-	��^�B=�A�*


acc	�>���       QKD	�]j�B=�A�*

val_loss�/?>���       ��2	_`j�B=�A�*

val_acc��>���a       �	�aj�B=�A�*

loss�.?Z�l       ��-	�bj�B=�A�*


acc�{�>��n       QKD	/�q�B=�A�*

val_loss�/?�j�       ��2	+�q�B=�A�*

val_accN�>R��       �	C�q�B=�A�*

lossY�.?��ʁ       ��-	�q�B=�A�*


accb�>��=5       QKD	X9|�B=�A�*

val_loss��/?y?ݶ       ��2	G:|�B=�A�*

val_acc,0�>���       �	�:|�B=�A�*

lossY/?A1�[       ��-	;|�B=�A�*


acc��>��       QKD	�v��B=�A�*

val_loss9�0?�Wݑ       ��2	�w��B=�A�*

val_acc:��>od2C       �	Rx��B=�A�*

loss�3/?(���       ��-	�x��B=�A�*


accw�>��+I       QKD	�珇B=�A�*

val_loss�H/?�_v�       ��2	m鏇B=�A�*

val_acc���>��2       �	�鏇B=�A�*

loss�9/?�SD�       ��-	dꏇB=�A�*


acc΋�>����       QKD	����B=�A�*

val_loss�A0?�|O�       ��2	����B=�A�*

val_acc%�>�^3�       �	7���B=�A�*

loss��.?�ڈ�       ��-	D���B=�A�*


acc���>gT�       QKD	Z���B=�A�*

val_loss�0?pSi_       ��2	����B=�A�*

val_accUB�>�܃       �	o���B=�A�*

loss��.?���       ��-	���B=�A�*


accEϑ>c,\	       QKD	���B=�A�*

val_loss��/?Q�       ��2	
��B=�A�*

val_acc��>c�)       �	&��B=�A�*

loss�/?^T�
       ��-	+��B=�A�*


acc�,�>���       QKD	ű�B=�A�*

val_loss�/?���       ��2	�Ʊ�B=�A�*

val_accc��>)i��       �	~Ǳ�B=�A�*

loss2�.?�F��       ��-	�Ǳ�B=�A�*


acc��>Yu�	