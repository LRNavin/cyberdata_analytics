       �K"	  ��B=�Abrain.Event:2���Z�     ��$	9��B=�A"��
j
input_1Placeholder*
dtype0*'
_output_shapes
:���������+*
shape:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *qĜ>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*

seed*
T0*
dtype0*
_output_shapes

:+*
seed2끞
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes

:+*
T0
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:+*
	container *
shape
:+
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+
{
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes

:+*
T0*!
_class
loc:@dense_1/kernel
Z
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_1/bias
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������
g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������
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
:���������
s
"dense_1/activity_regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
valueB"   
   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *�?�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *�?�>*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes

:
*
seed2�ߜ*

seed
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:

~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes

:
*
T0
�
dense_2/kernel
VariableV2*
dtype0*
_output_shapes

:
*
	container *
shape
:
*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(*
_output_shapes

:
*
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

:

Z
dense_2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_2/bias
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
*
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
T0*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

W
dense_2/ReluReludense_2/BiasAdd*'
_output_shapes
:���������
*
T0
m
dense_3/random_uniform/shapeConst*
valueB"
   
   *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *�7�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�7?
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes

:

*
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

:


~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:


�
dense_3/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:

*
	container *
shape
:


�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:


{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:


Z
dense_3/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_3/bias
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
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
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
:���������
*
transpose_a( *
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:���������

m
dense_4/random_uniform/shapeConst*
valueB"
   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
_output_shapes

:
+*
seed2�Ֆ*

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

:
+
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:
+*
T0
�
dense_4/kernel
VariableV2*
dtype0*
_output_shapes

:
+*
	container *
shape
:
+*
shared_name 
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:
+*
use_locking(
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:
+
Z
dense_4/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
x
dense_4/bias
VariableV2*
shape:+*
shared_name *
dtype0*
_output_shapes
:+*
	container 
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
dense_4/bias/readIdentitydense_4/bias*
_output_shapes
:+*
T0*
_class
loc:@dense_4/bias
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
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(
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
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(
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
dense_4_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
l
loss/dense_4_loss/subSubdense_4/Reludense_4_target*'
_output_shapes
:���������+*
T0
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
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
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
loss/mul/xloss/dense_4_loss/Mean_3*
_output_shapes
: *
T0
\
loss/addAddloss/mul dense_1/activity_regularizer/add*
T0*
_output_shapes
: 
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
metrics/acc/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:���������
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
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
}
training/Adam/gradients/ShapeConst*
valueB *
_class
loc:@loss/add*
dtype0*
_output_shapes
: 
�
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?*
_class
loc:@loss/add
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0*
_class
loc:@loss/add
�
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_3*
_class
loc:@loss/mul*
_output_shapes
: *
T0
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
:*
valueB:*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
:
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
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB *+
_class!
loc:@loss/dense_4_loss/Mean_3
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
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
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordiv*

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Cast*#
_output_shapes
:���������*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3
�
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*
valueB"      *3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
dtype0*
_output_shapes
:
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
T0*
out_type0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
_output_shapes
:
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*'
_output_shapes
:���������
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *,
_class"
 loc:@loss/dense_4_loss/truediv
�
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:���������:���������
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/truedivloss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
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
:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*'
_output_shapes
:���������*
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
:���������*
T0*
Tshape0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
out_type0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*
T0
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
:*

Tidx0*
	keep_dims( 
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
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
:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*'
_output_shapes
:���������*
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
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
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
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/FillFill=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/value*
T0*

index_type0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
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
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������*
T0
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Reshape>training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv*#
_output_shapes
:���������*

Tmultiples0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
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
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0
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
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/CastCast@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1*
_output_shapes
: *

DstT0*

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
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *)
_class
loc:@loss/dense_4_loss/Mean
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
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*

index_type0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0
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
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2Shapeloss/dense_4_loss/Square*
_output_shapes
:*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean
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
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
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
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*'
_output_shapes
:���������+*
T0*)
_class
loc:@loss/dense_4_loss/Mean
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
9training/Adam/gradients/loss/dense_4_loss/Square_grad/MulMulloss/dense_4_loss/sub;training/Adam/gradients/loss/dense_4_loss/Square_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
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
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
_output_shapes
:*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/sub
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
:*

Tidx0*
	keep_dims( 
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
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
T0
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
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
_output_shapes

:
+*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*'
_output_shapes
:���������
*
T0*
_class
loc:@dense_3/Tanh
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:

�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:

*
transpose_a(*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������

�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
_output_shapes
:
*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:
*
transpose_a(
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*'
_output_shapes
:���������+*
transpose_a( *
transpose_b(*
T0*!
_class
loc:@dense_1/MatMul
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
_output_shapes

:+*
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
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
p
training/Adam/CastCastAdam/iterations/read*
_output_shapes
: *

DstT0*

SrcT0	*
Truncate( 
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
training/Adam/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
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
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable
VariableV2*
_output_shapes

:+*
	container *
shape
:+*
shared_name *
dtype0
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+
b
training/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:
j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes

:
*
	container *
shape
:

�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
_output_shapes

:
*
T0*+
_class!
loc:@training/Adam/Variable_2
b
training/Adam/zeros_3Const*
dtype0*
_output_shapes
:
*
valueB
*    
�
training/Adam/Variable_3
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
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
*
use_locking(
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:
*
T0
j
training/Adam/zeros_4Const*
_output_shapes

:

*
valueB

*    *
dtype0
�
training/Adam/Variable_4
VariableV2*
shared_name *
dtype0*
_output_shapes

:

*
	container *
shape
:


�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:


�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:


b
training/Adam/zeros_5Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_5
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:

�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_5
j
training/Adam/zeros_6Const*
valueB
+*    *
dtype0*
_output_shapes

:
+
�
training/Adam/Variable_6
VariableV2*
dtype0*
_output_shapes

:
+*
	container *
shape
:
+*
shared_name 
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:
+
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:
+
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
j
training/Adam/zeros_8Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_8
VariableV2*
shape
:+*
shared_name *
dtype0*
_output_shapes

:+*
	container 
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+
b
training/Adam/zeros_9Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_9
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_9
k
training/Adam/zeros_10Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_10
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:

c
training/Adam/zeros_11Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
_output_shapes
:
*
	container *
shape:

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

k
training/Adam/zeros_12Const*
dtype0*
_output_shapes

:

*
valueB

*    
�
training/Adam/Variable_12
VariableV2*
dtype0*
_output_shapes

:

*
	container *
shape
:

*
shared_name 
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:

*
use_locking(
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:


c
training/Adam/zeros_13Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_13
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
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:

�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes
:
*
T0*,
_class"
 loc:@training/Adam/Variable_13
k
training/Adam/zeros_14Const*
dtype0*
_output_shapes

:
+*
valueB
+*    
�
training/Adam/Variable_14
VariableV2*
dtype0*
_output_shapes

:
+*
	container *
shape
:
+*
shared_name 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:
+*
use_locking(*
T0
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:
+
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
dtype0*
_output_shapes
:+*
	container 
�
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0
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
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*
_output_shapes
:
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
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_16
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
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_17
p
&training/Adam/zeros_18/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_18/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
&training/Adam/zeros_19/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
T0*,
_class"
 loc:@training/Adam/Variable_20*
_output_shapes
:
p
&training/Adam/zeros_21/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes
:*
T0
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
&training/Adam/zeros_23/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_23/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
dtype0*
_output_shapes
:*
	container 
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

:+
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

:+
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes

:+*
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
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:+
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes

:+*
T0
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+
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

:+
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
_output_shapes

:+*
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

:+
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes

:+
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:+
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:
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
:*
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes
:*
T0
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes
:*
T0
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
:
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:
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
:
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
:*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:
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
:
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:*
T0
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
_output_shapes
:*
T0
�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:

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

:

o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:

v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
_output_shapes

:
*
T0
Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:

l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:

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

:

�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
_output_shapes

:
*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:

Z
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
p
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
_output_shapes

:
*
T0
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
_output_shapes

:
*
T0
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:

�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2
�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:

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
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
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
training/Adam/Const_8Const*
_output_shapes
: *
valueB
 *    *
dtype0
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
:

�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
:
*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:

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
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:

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
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:


[
training/Adam/sub_14/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:


p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:


v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:


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

:


r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
_output_shapes

:

*
T0
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*
_output_shapes

:


m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:


[
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *    
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

:


�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:


d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:


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

:

*
T0
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:


r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:


�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
_output_shapes

:

*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:

*
use_locking(
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:


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
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:

l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:
*
T0
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
training/Adam/Const_13Const*
dtype0*
_output_shapes
: *
valueB
 *  �
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
training/Adam/add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *���3
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:

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
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
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
:
*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:
+
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

:
+
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:
+
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
_output_shapes

:
+*
T0
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

:
+
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes

:
+*
T0
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:
+
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:
+
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

:
+*
T0
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:
+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:
+
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

:
+*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes

:
+*
T0
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

:
+*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:
+*
use_locking(
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:
+
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:
+
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
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 
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
training/Adam/Const_17Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
_output_shapes
:+*
T0
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
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes
:+*
T0
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:+
l
training/Adam/sub_25Subdense_4/bias/readtraining/Adam/truediv_8*
_output_shapes
:+*
T0
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
IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_output_shapes
: *
_class
loc:@dense_1/bias*
dtype0
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
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0
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
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_18*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_18*
dtype0
�
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_19*,
_class"
 loc:@training/Adam/Variable_19*
dtype0*
_output_shapes
: 
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
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"!�>�*�     �:S[	����B=�AJ��
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
valueB"+      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *qĜ>*
dtype0
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*

seed*
T0*
dtype0*
seed2끞*
_output_shapes

:+
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:+
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+*
use_locking(
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+
Z
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_1/bias
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������
g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������
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
:���������
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
valueB"   
   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *�?�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *�?�>*
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

:

z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes

:
*
T0
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:

�
dense_2/kernel
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:

{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:

Z
dense_2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*    
x
dense_2/bias
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
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
_output_shapes
:
*
T0*
_class
loc:@dense_2/bias
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
transpose_a( *'
_output_shapes
:���������
*
transpose_b( *
T0
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
valueB"
   
   *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
_output_shapes
: *
valueB
 *�7�*
dtype0
_
dense_3/random_uniform/maxConst*
valueB
 *�7?*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
seed2��*
_output_shapes

:

*

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

:


~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:

*
T0
�
dense_3/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:

*
shape
:


�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:

*
use_locking(*
T0
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:


Z
dense_3/ConstConst*
_output_shapes
:
*
valueB
*    *
dtype0
x
dense_3/bias
VariableV2*
shape:
*
shared_name *
dtype0*
	container *
_output_shapes
:

�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:

�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:���������

�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������

W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:���������

m
dense_4/random_uniform/shapeConst*
valueB"
   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�D�>
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
seed2�Ֆ*
_output_shapes

:
+*

seed
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

:
+*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:
+
�
dense_4/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:
+*
shape
:
+*
shared_name 
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:
+
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:
+
Z
dense_4/ConstConst*
dtype0*
_output_shapes
:+*
valueB+*    
x
dense_4/bias
VariableV2*
shape:+*
shared_name *
dtype0*
	container *
_output_shapes
:+
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0
q
dense_4/bias/readIdentitydense_4/bias*
_output_shapes
:+*
T0*
_class
loc:@dense_4/bias
�
dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������+*
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
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
shape: 
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
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
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
loss/dense_4_loss/subSubdense_4/Reludense_4_target*'
_output_shapes
:���������+*
T0
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
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
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
loss/dense_4_loss/CastCastloss/dense_4_loss/NotEqual*

DstT0*#
_output_shapes
:���������*

SrcT0
*
Truncate( 
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
loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_2*#
_output_shapes
:���������*
T0
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

loss/mul/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
metrics/acc/ArgMaxArgMaxdense_4_targetmetrics/acc/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
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
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:���������
[
metrics/acc/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_output_shapes
: *
T0*
_class
loc:@loss/add*

index_type0
�
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_3*
_output_shapes
: *
T0*
_class
loc:@loss/mul
�
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss/mul
�
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB:*
dtype0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Tshape0*
_output_shapes
:
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
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1Shapeloss/dense_4_loss/truediv*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0*
_output_shapes
:
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
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: 
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
?training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum/yConst*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
value	B :*
dtype0
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

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Truncate( 
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
:*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
valueB"      
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
Tshape0*
_output_shapes

:
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
out_type0*
_output_shapes
:
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*'
_output_shapes
:���������
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
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:���������*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0
�
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:���������*
T0
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
:*
	keep_dims( *

Tidx0
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
_output_shapes
: *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ShapeConst*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
valueB *
dtype0*
_output_shapes
: 
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
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulMulBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile dense_1/activity_regularizer/Abs*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*
_output_shapes
: 
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������
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
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0*
_output_shapes
:
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
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
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
:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*'
_output_shapes
:���������*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ShapeShapeloss/dense_4_loss/Mean*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:*
T0
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
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
:*

Tidx0
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill/valueConst*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :*
dtype0
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2Shapeloss/dense_4_loss/Mean*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: 
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
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
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeloss/dense_4_loss/Square*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:
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
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*

Tidx0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
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
<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
Tshape0
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
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:
�
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
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
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *

DstT0*
_output_shapes
: 
�
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*
T0*)
_class
loc:@loss/dense_4_loss/Mean*'
_output_shapes
:���������+
�
;training/Adam/gradients/loss/dense_4_loss/Square_grad/ConstConst<^training/Adam/gradients/loss/dense_4_loss/Mean_grad/truediv*+
_class!
loc:@loss/dense_4_loss/Square*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
9training/Adam/gradients/loss/dense_4_loss/Square_grad/MulMulloss/dense_4_loss/sub;training/Adam/gradients/loss/dense_4_loss/Square_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Square*'
_output_shapes
:���������+
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
:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1Shapedense_4_target*
T0*(
_class
loc:@loss/dense_4_loss/sub*
out_type0*
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
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
transpose_a( *'
_output_shapes
:���������
*
transpose_b(*
T0*!
_class
loc:@dense_4/MatMul
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_4/MatMul*
transpose_a(*
_output_shapes

:
+*
transpose_b( 
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
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( *'
_output_shapes
:���������

�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*!
_class
loc:@dense_3/MatMul*
transpose_a(*
_output_shapes

:

*
transpose_b( *
T0
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*'
_output_shapes
:���������
*
T0*
_class
loc:@dense_2/Relu
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
:���������*
transpose_b(*
T0
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(*
_output_shapes

:
*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*'
_output_shapes
:���������*
T0*
_class
loc:@dense_1/Tanh
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:
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

:+*
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
training/Adam/CastCastAdam/iterations/read*
Truncate( *

DstT0*
_output_shapes
: *

SrcT0	
X
training/Adam/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
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
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  �
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0
�
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
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
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
h
training/Adam/zerosConst*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
_output_shapes

:+*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+
b
training/Adam/zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:
j
training/Adam/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_2
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:

b
training/Adam/zeros_3Const*
dtype0*
_output_shapes
:
*
valueB
*    
�
training/Adam/Variable_3
VariableV2*
shape:
*
shared_name *
dtype0*
	container *
_output_shapes
:

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
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_3
j
training/Adam/zeros_4Const*
valueB

*    *
dtype0*
_output_shapes

:


�
training/Adam/Variable_4
VariableV2*
dtype0*
	container *
_output_shapes

:

*
shape
:

*
shared_name 
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:


�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes

:


b
training/Adam/zeros_5Const*
_output_shapes
:
*
valueB
*    *
dtype0
�
training/Adam/Variable_5
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
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
*
use_locking(
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_5
j
training/Adam/zeros_6Const*
valueB
+*    *
dtype0*
_output_shapes

:
+
�
training/Adam/Variable_6
VariableV2*
dtype0*
	container *
_output_shapes

:
+*
shape
:
+*
shared_name 
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:
+*
use_locking(*
T0
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes

:
+*
T0*+
_class!
loc:@training/Adam/Variable_6
b
training/Adam/zeros_7Const*
dtype0*
_output_shapes
:+*
valueB+*    
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
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:+*
use_locking(
�
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:+
j
training/Adam/zeros_8Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_8
VariableV2*
shape
:+*
shared_name *
dtype0*
	container *
_output_shapes

:+
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
_output_shapes

:+*
T0*+
_class!
loc:@training/Adam/Variable_8
b
training/Adam/zeros_9Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_9
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:
k
training/Adam/zeros_10Const*
valueB
*    *
dtype0*
_output_shapes

:

�
training/Adam/Variable_10
VariableV2*
dtype0*
	container *
_output_shapes

:
*
shape
:
*
shared_name 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
_output_shapes

:
*
T0*,
_class"
 loc:@training/Adam/Variable_10
c
training/Adam/zeros_11Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_11
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
:
*
T0*,
_class"
 loc:@training/Adam/Variable_11
k
training/Adam/zeros_12Const*
valueB

*    *
dtype0*
_output_shapes

:


�
training/Adam/Variable_12
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:

*
shape
:


�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
validate_shape(*
_output_shapes

:

*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:


c
training/Adam/zeros_13Const*
valueB
*    *
dtype0*
_output_shapes
:

�
training/Adam/Variable_13
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
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes
:
*
T0*,
_class"
 loc:@training/Adam/Variable_13
k
training/Adam/zeros_14Const*
valueB
+*    *
dtype0*
_output_shapes

:
+
�
training/Adam/Variable_14
VariableV2*
dtype0*
	container *
_output_shapes

:
+*
shape
:
+*
shared_name 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
validate_shape(*
_output_shapes

:
+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:
+
c
training/Adam/zeros_15Const*
dtype0*
_output_shapes
:+*
valueB+*    
�
training/Adam/Variable_15
VariableV2*
	container *
_output_shapes
:+*
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
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:+
p
&training/Adam/zeros_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(
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
training/Adam/zeros_19/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_19
p
&training/Adam/zeros_20/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_20
p
&training/Adam/zeros_21/shape_as_tensorConst*
_output_shapes
:*
valueB:*
dtype0
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
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_22
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
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*

index_type0*
_output_shapes
:*
T0
�
training/Adam/Variable_23
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
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

:+
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

:+
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes

:+*
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

:+*
T0
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:+
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes

:+
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+
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

:+
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
_output_shapes

:+*
T0
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:+
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

:+
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

:+*
T0
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
_output_shapes

:+*
T0
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable
�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
:*
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
:
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes
:*
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
:*
T0
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
:
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes
:*
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
:
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
:*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:*
T0
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
:
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
:
�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:

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
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
_output_shapes

:
*
T0
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:

Z
training/Adam/sub_9/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:

q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:

o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
_output_shapes

:
*
T0
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:

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

:

�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes

:

d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:

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

:

v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:

r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:

�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:

�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:

q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:

[
training/Adam/sub_11/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
:
*
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
:

�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:

`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:

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
:
*
T0
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
_output_shapes
:
*
T0
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
*
use_locking(
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:

u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:


[
training/Adam/sub_14/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:

*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:


v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:


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

:


r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
_output_shapes

:

*
T0
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
_output_shapes

:

*
T0
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
_output_shapes

:

*
T0
[
training/Adam/Const_10Const*
_output_shapes
: *
valueB
 *    *
dtype0
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

:


�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:


d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*
_output_shapes

:


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

:


w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:


r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:


�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:


�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:

*
use_locking(
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:

*
use_locking(*
T0
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
training/Adam/sub_18/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
:
*
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:
*
T0
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
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes
:
*
T0
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
:
*
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

:
+
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
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:
+*
T0
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:
+
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:
+
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
training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:
+*
T0
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:
+
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:
+
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
_output_shapes

:
+*
T0
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*
_output_shapes

:
+
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:
+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes

:
+*
T0
[
training/Adam/add_21/yConst*
_output_shapes
: *
valueB
 *���3*
dtype0
r
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes

:
+
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes

:
+*
T0
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes

:
+
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:
+
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:
+*
use_locking(
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:
+*
use_locking(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:+
[
training/Adam/sub_23/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
training/Adam/Const_16Const*
dtype0*
_output_shapes
: *
valueB
 *    
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
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_output_shapes
: *
_class
loc:@dense_4/bias*
dtype0
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
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0
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
IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4
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
IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10*
dtype0
�
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 
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
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08ɛ�C       ���	hz��B=�A*

val_loss2�?:Kz�       �	|��B=�A*

val_accR��=M�v       �K"	�|��B=�A*

lossC�@S��<       ���	[}��B=�A*


acc-�%=�e�g       ��2	�'��B=�A*

val_loss$G�?���4       `/�#	g)��B=�A*

val_acc
�=˸hl       ��-	�)��B=�A*

lossmӾ?h=C�       ��(	E*��B=�A*


acc��=C��       ��2	x���B=�A*

val_loss�x?Ile�       `/�#	����B=�A*

val_acc\�=��H       ��-	����B=�A*

loss<g�?=4J4       ��(	V���B=�A*


accv#�=h=|       ��2	����B=�A*

val_loss��_?��w       `/�#	#���B=�A*

val_acc��=d���       ��-	����B=�A*

loss��s?B��       ��(	{���B=�A*


accƥ�=�+��       ��2	����B=�A*

val_loss�|Q?�       `/�#	R���B=�A*

val_acc
�=V�[�       ��-	���B=�A*

loss)�`?Хfa       ��(	����B=�A*


accce�=I~ӊ       ��2	B��B=�A*

val_loss�9I?R�Q       `/�#	8��B=�A*

val_acc\�=���[       ��-	^��B=�A*

loss�`U?���       ��(	���B=�A*


acc���=0�l�       ��2	~�B=�A*

val_loss�<D?���       `/�#	��B=�A*

val_accx��=�>Z       ��-	ǀ�B=�A*

loss[�N?       ��(	���B=�A*


acc�c�=�       ��2	\��B=�A*

val_loss*@?�GƬ       `/�#	��B=�A*

val_acc��=��u       ��-	���B=�A*

loss�J?g�R       ��(	��B=�A*


acc��=~��       ��2	���B=�A*

val_loss�]<?�øO       `/�#	��B=�A*

val_acc��=x���       ��-	���B=�A*

loss�mE?���       ��(	Z��B=�A*


acc���=��w
       ��2	�#�B=�A	*

val_lossaQ9?)��       `/�#	��#�B=�A	*

val_acc�9�=�6��       ��-	~�#�B=�A	*

loss"lA?N���       ��(	�#�B=�A	*


acc�)�=�dMS       ��2	-�/�B=�A
*

val_lossl�6?��]�       `/�#	ɳ/�B=�A
*

val_accO��=�s�       ��-	��/�B=�A
*

lossFE>?X��L       ��(	�/�B=�A
*


accJ��=��h�       ��2	�3;�B=�A*

val_loss�4?�qJ       `/�#	a5;�B=�A*

val_acc��D>�ѾE       ��-	6;�B=�A*

loss/ ;?	�f       ��(	�6;�B=�A*


accx�>P;��       ��2	�NG�B=�A*

val_loss�s1?ʻ5�       `/�#	*PG�B=�A*

val_accďj>���       ��-	�PG�B=�A*

loss�7?����       ��(	QG�B=�A*


acc�O[>��.?       ��2	30S�B=�A*

val_loss�/?#��       `/�#	�1S�B=�A*

val_acc<~>�Y�~       ��-	�2S�B=�A*

loss�4?����       ��(	+3S�B=�A*


acc��r>F���       ��2	=�^�B=�A*

val_loss�#-?ߚW"       `/�#	Ԝ^�B=�A*

val_accȑ>�`[�       ��-	��^�B=�A*

loss��2?�q؋       ��(	(�^�B=�A*


acc�&�>��Z�       ��2	Lrj�B=�A*

val_lossZ�+?�g+�       `/�#	~sj�B=�A*

val_acc�̓>�2q�       ��-	�sj�B=�A*

loss�E0?`�:       ��(	atj�B=�A*


acc��>Fo}       ��2	��u�B=�A*

val_loss��(?�ƉS       `/�#	��u�B=�A*

val_acc�,�>���       ��-	r�u�B=�A*

loss��-?	�	�       ��(	�u�B=�A*


acc�,�>%�v;       ��2	���B=�A*

val_loss#�%?��`       `/�#	����B=�A*

val_acc«�>ߛ�*       ��-	c���B=�A*

lossv�*?�Q[y       ��(	����B=�A*


acc���>�!�D       ��2	���B=�A*

val_loss��#?J`��       `/�#	b��B=�A*

val_acc�N�>�q�,       ��-	$��B=�A*

loss�,(?yk�&       ��(	���B=�A*


acc�T�>���       ��2	�m��B=�A*

val_loss+�"?9,8       `/�#	2o��B=�A*

val_acc��>�eY       ��-	�o��B=�A*

loss�&?DȄ
       ��(	�p��B=�A*


acc��>��w&       ��2	�@��B=�A*

val_loss{�!?���       `/�#	�B��B=�A*

val_acc r�>WgΉ       ��-	�C��B=�A*

loss�W%?)��e       ��(	5D��B=�A*


acc*��>o��       ��2	^��B=�A*

val_lossT!?X��g       `/�#	�_��B=�A*

val_accg��>6M&�       ��-	y`��B=�A*

loss� $?�M�       ��(	a��B=�A*


acc�;�>{ҥ4       ��2	|ػ�B=�A*

val_lossp0 ?���       `/�#	ڻ�B=�A*

val_acc_��>=�X�       ��-	�ڻ�B=�A*

loss�#?��
       ��(	jۻ�B=�A*


accpS�>�ˇ       ��2	Bx��B=�A*

val_loss>c?Nw\       `/�#	ly��B=�A*

val_acc�g�>�5�       ��-	�y��B=�A*

lossd5"?��Y       ��(	Wz��B=�A*


accm��>���       ��2	3k��B=�A*

val_loss��?��Z       `/�#	�l��B=�A*

val_acc��>~81�       ��-	~m��B=�A*

loss/X!?:�       ��(	n��B=�A*


accbd�>L�       ��2	U��B=�A*

val_loss}?Vώ�       `/�#	��B=�A*

val_acc�K�>or��       ��-	���B=�A*

loss� ?Z8�W       ��(	���B=�A*


acc!�>��       ��2	n0��B=�A*

val_loss�5?�o�9       `/�#	82��B=�A*

val_acc��>�0�@       ��-	�2��B=�A*

loss�?5��       ��(	<3��B=�A*


acc=��>+��       ��2	����B=�A*

val_lossZ�?��        `/�#	T���B=�A*

val_acc4P�>���Y       ��-	���B=�A*

loss��?z�:R       ��(	����B=�A*


acc/>�>n eT       ��2	Dq�B=�A*

val_loss�?2ǖ       `/�#	�r�B=�A*

val_acc3��>�3�S       ��-	�s�B=�A*

lossw$?o���       ��(	Bt�B=�A*


acc��>�@�       ��2	�	�B=�A*

val_lossO�?���u       `/�#	x�B=�A*

val_accN?�>��k       ��-	5�B=�A*

loss,o?�^@w       ��(	��B=�A*


accE�>q��;       ��2	���B=�A*

val_lossJ?�N4�       `/�#	��B=�A*

val_acc�J�>
 �m       ��-	f��B=�A*

loss��?3�       ��(	˃�B=�A*


acc��>T       ��2	v$�B=�A*

val_loss?�R�       `/�#	w$�B=�A*

val_acc�3�>nzW       ��-	}w$�B=�A*

loss��?�J�q       ��(	�w$�B=�A*


acc�
�>ɐ�H       ��2	�	1�B=�A *

val_loss�!?��Q       `/�#	W1�B=�A *

val_accT%�>�K��       ��-	1�B=�A *

loss��?����       ��(	�1�B=�A *


acc+��>D�       ��2	��9�B=�A!*

val_loss�D?�NO.       `/�#	�9�B=�A!*

val_acc���>�&6       ��-	g�9�B=�A!*

lossA�?,�       ��(	ϻ9�B=�A!*


acc��>=t��       ��2	WzD�B=�A"*

val_loss��?(�s�       `/�#	�{D�B=�A"*

val_accE?��d       ��-	|D�B=�A"*

lossE�?���&       ��(	x|D�B=�A"*


acciD ?��(       ��2	��L�B=�A#*

val_loss��?[T�j       `/�#	��L�B=�A#*

val_acc�i?DA}A       ��-	>�L�B=�A#*

loss?d��       ��(	��L�B=�A#*


acc��?V���       ��2	�kU�B=�A$*

val_loss<�?9�       `/�#	�lU�B=�A$*

val_acc�` ?��p�       ��-	7mU�B=�A$*

loss�q?f��       ��(	�mU�B=�A$*


acc� ?���       ��2	��_�B=�A%*

val_loss�?��X4       `/�#	;�_�B=�A%*

val_acc�n�>DO�       ��-	��_�B=�A%*

loss��?�]�       ��(	&�_�B=�A%*


acc�@?7(ִ       ��2	dsj�B=�A&*

val_loss�m?D�S       `/�#	uj�B=�A&*

val_accŬ?��0?       ��-	�uj�B=�A&*

lossT�?c},�       ��(	�uj�B=�A&*


accx� ?�q�       ��2	-�u�B=�A'*

val_loss?I?y���       `/�#	Ǚu�B=�A'*

val_acc�I?����       ��-	��u�B=�A'*

lossT?��W�       ��(	 �u�B=�A'*


acc�?����       ��2	]���B=�A(*

val_loss �?����       `/�#	����B=�A(*

val_acc�?U�w"       ��-	.���B=�A(*

loss�?�R A       ��(	⫀�B=�A(*


accH?���
       ��2	���B=�A)*

val_lossλ?5�Z�       `/�#	,��B=�A)*

val_acc�+?�+       ��-	���B=�A)*

lossk�?�3       ��(	���B=�A)*


acc<?[b^�       ��2	8���B=�A**

val_lossv�?#�*       `/�#	����B=�A**

val_acc��?Y��.       ��-	a���B=�A**

lossɡ?T6�       ��(	����B=�A**


acc��?Q3�       ��2	�؞�B=�A+*

val_loss9�?���       `/�#	�ٞ�B=�A+*

val_acc��?I�D       ��-	ڞ�B=�A+*

loss%,?Z�+       ��(	ڞ�B=�A+*


acc�=?�4�       ��2	u��B=�A,*

val_lossǄ?t��#       `/�#		��B=�A,*

val_acc��?���       ��-	�	��B=�A,*

loss+�?}=~�       ��(	y
��B=�A,*


acc��?��*
       ��2	���B=�A-*

val_loss�?V��;       `/�#	����B=�A-*

val_acc�n�>ka�       ��-	f���B=�A-*

loss�?6@h�       ��(	����B=�A-*


acc��?��\       ��2	���B=�A.*

val_loss�?L��*       `/�#	����B=�A.*

val_acc*��>׳�E       ��-	h���B=�A.*

loss��?	��P       ��(	���B=�A.*


acc /?X���       ��2	����B=�A/*

val_loss�?U       `/�#	����B=�A/*

val_acc�|�>��z�       ��-	]���B=�A/*

loss6I?z�8�       ��(	����B=�A/*


acc2	?��'       ��2	|���B=�A0*

val_lossl�?��c       `/�#	���B=�A0*

val_acc�+?���       ��-	؝��B=�A0*

lossG?H��       ��(	o���B=�A0*


acců?o���       ��2	;S��B=�A1*

val_loss�w?sA��       `/�#	�T��B=�A1*

val_acc�
?�w�       ��-	�U��B=�A1*

loss�?�^Q       ��(	:V��B=�A1*


acc�?%���       ��2	
���B=�A2*

val_loss�j?��W       `/�#	r���B=�A2*

val_acc��?��       ��-	<���B=�A2*

lossz?�e�       ��(	ۢ��B=�A2*


acc�&?S1�       ��2	����B=�A3*

val_loss�q?�Ό�       `/�#	����B=�A3*

val_acc|?*�dy       ��-	���B=�A3*

loss	[?�O�Y       ��(	���B=�A3*


acc`�?��       ��2	X��B=�A4*

val_lossau?;�       `/�#	���B=�A4*

val_acc�
?0N��       ��-	X��B=�A4*

loss9�?�_��       ��(	���B=�A4*


accS�?渲�       ��2	5��B=�A5*

val_loss��?ޜp2       `/�#	1��B=�A5*

val_acc��?����       ��-	��B=�A5*

loss?�?���m       ��(	���B=�A5*


accd�
?ǘ��       ��2	��B=�A6*

val_lossI?�<#       `/�#	r�B=�A6*

val_acc�?D:��       ��-	4�B=�A6*

loss�.?S�^       ��(	��B=�A6*


accE�?�	\�       ��2	b��B=�A7*

val_loss:?.�++       `/�#	��B=�A7*

val_acc��?��       ��-	���B=�A7*

loss�?�w�       ��(	^��B=�A7*


acc�.?����       ��2	_"�B=�A8*

val_loss��?�]�       `/�#	�`"�B=�A8*

val_acc,0?V�$�       ��-	9a"�B=�A8*

loss��?�5       ��(	�a"�B=�A8*


acc
o?��:�       ��2	x�+�B=�A9*

val_loss,�?$�(�       `/�#	y�+�B=�A9*

val_acc��?���       ��-	��+�B=�A9*

loss+�?Dx|�       ��(	g�+�B=�A9*


acc�[?��3�       ��2	C�6�B=�A:*

val_loss:O?\�4       `/�#	��6�B=�A:*

val_acc�?���       ��-	e�6�B=�A:*

loss�y?R��       ��(	��6�B=�A:*


acc�?��u�       ��2	u�A�B=�A;*

val_lossN?���       `/�#	��A�B=�A;*

val_accY`?�&�L       ��-	�A�B=�A;*

loss�??��w       ��(	p�A�B=�A;*


acc/�?��T        ��2	��L�B=�A<*

val_loss2?pW9Q       `/�#	��L�B=�A<*

val_acc�w
?r�<E       ��-	/�L�B=�A<*

loss�/?po*       ��(	��L�B=�A<*


acc�?�5��       ��2	V+U�B=�A=*

val_loss�m?��c�       `/�#	�,U�B=�A=*

val_acc��?Q�d       ��-	9-U�B=�A=*

loss��?�,Ua       ��(	�-U�B=�A=*


acc��?~j�       ��2	��]�B=�A>*

val_loss�?�W       `/�#	��]�B=�A>*

val_acc�p?�B�`       ��-	,�]�B=�A>*

loss�?����       ��(	��]�B=�A>*


accr?܄N�       ��2	��h�B=�A?*

val_loss��?T>��       `/�#	$�h�B=�A?*

val_acc\~?#p��       ��-	��h�B=�A?*

loss'�?���t       ��(	�h�B=�A?*


acc�x?��U       ��2	��p�B=�A@*

val_loss��?2�-�       `/�#	Z�p�B=�A@*

val_acc(g?s��       ��-	5�p�B=�A@*

loss�?Ê�        ��(	��p�B=�A@*


acc\�?E�s"       ��2	p{�B=�AA*

val_loss��?P��       `/�#	�q{�B=�AA*

val_acc@?Ar9�       ��-	\r{�B=�AA*

loss>�?�G��       ��(	�r{�B=�AA*


acc��?���e       ��2	ڐ��B=�AB*

val_loss7�?��       `/�#	K���B=�AB*

val_acc�?\�W       ��-	���B=�AB*

loss��?V�e"       ��(	����B=�AB*


acc��?TD-l       ��2	�g��B=�AC*

val_lossra?C�u       `/�#	�i��B=�AC*

val_acc�Y?�T�       ��-	bj��B=�AC*

loss�?���%       ��(	�j��B=�AC*


acc7�?Sa.�       ��2	e;��B=�AD*

val_loss�%?EN�       `/�#	�<��B=�AD*

val_acc��
?���i       ��-	�=��B=�AD*

loss�#?��>K       ��(	X>��B=�AD*


acc.#?��HE       ��2	+T��B=�AE*

val_lossb�?N+�l       `/�#	�U��B=�AE*

val_acc�R?d�P       ��-	eV��B=�AE*

loss�?���       ��(	�V��B=�AE*


acc��?���       ��2	�u��B=�AF*

val_loss�?-�       `/�#	.w��B=�AF*

val_accc�?9^�=       ��-	�w��B=�AF*

loss��?���4       ��(	}x��B=�AF*


accړ?m�]       ��2	�W��B=�AG*

val_loss>�?"���       `/�#	�X��B=�AG*

val_acc�d?':��       ��-	�Y��B=�AG*

loss�?��nI       ��(	KZ��B=�AG*


acc�s?@J       ��2	R���B=�AH*

val_loss��?5,xE       `/�#	F���B=�AH*

val_accD�?şK�       ��-	����B=�AH*

loss ?��/;       ��(	 ���B=�AH*


acc�!?����       ��2	N���B=�AI*

val_loss��?£�       `/�#	����B=�AI*

val_acc�?��       ��-	����B=�AI*

lossB?�yU       ��(	����B=�AI*


acc=�?j���       ��2	�W��B=�AJ*

val_loss�q?%ªE       `/�#	�X��B=�AJ*

val_acc��>�1�~       ��-	BY��B=�AJ*

loss�?O�%�       ��(	�Y��B=�AJ*


acc�^?�F       ��2	0i��B=�AK*

val_loss�?�Kj       `/�#	�j��B=�AK*

val_acc6�?G�(K       ��-	�k��B=�AK*

loss�?͟�       ��(	+l��B=�AK*


acc\�?���       ��2	#��B=�AL*

val_lossl�?�Q       `/�#	�#��B=�AL*

val_acc��?���&       ��-	t$��B=�AL*

loss��?H�c       ��(	�$��B=�AL*


acc��?��       ��2	q��B=�AM*

val_loss��?��       `/�#	7r��B=�AM*

val_acc^?���       ��-	�r��B=�AM*

loss�M?�|       ��(	s��B=�AM*


acci�?��       ��2	����B=�AN*

val_lossW??�_@d       `/�#	����B=�AN*

val_acc:�?Kw��       ��-	3���B=�AN*

loss��?!O       ��(	����B=�AN*


accb�?�f�       ��2	���B=�AO*

val_loss�?@�e�       `/�#	��B=�AO*

val_acc�(?�\��       ��-	���B=�AO*

loss��?5&x�       ��(	���B=�AO*


acc�?U���       ��2	8.�B=�AP*

val_lossu�?�
�z       `/�#	�/�B=�AP*

val_acc�=?��       ��-	A0�B=�AP*

loss.?èŞ       ��(	�0�B=�AP*


acc�n?��       ��2	���B=�AQ*

val_loss�?���       `/�#	��B=�AQ*

val_acc�(?��I�       ��-	k��B=�AQ*

lossZ+?��h       ��(	ӂ�B=�AQ*


acce?��
A       ��2	��$�B=�AR*

val_loss�?~�,'       `/�#	1�$�B=�AR*

val_acc��?k.��       ��-	��$�B=�AR*

loss�?�×       ��(	�$�B=�AR*


acc�?pE       ��2	�/�B=�AS*

val_loss�l?r��       `/�#	i�/�B=�AS*

val_acc�?R%D�       ��-	"�/�B=�AS*

lossQ�?��       ��(	��/�B=�AS*


acc�?ł�F       ��2	98�B=�AT*

val_lossFa?8�c�       `/�#	(8�B=�AT*

val_acc^?)"-�       ��-	�8�B=�AT*

losso�?2d׭       ��(	�8�B=�AT*


acc�?��%�       ��2	ѕ@�B=�AU*

val_loss�?�6       `/�#	��@�B=�AU*

val_acc�?N�\+       ��-	:�@�B=�AU*

loss��?U!�       ��(	��@�B=�AU*


accHx?���       ��2	�I�B=�AV*

val_lossT�?^��       `/�#	�I�B=�AV*

val_accR�?�%"�       ��-	I�B=�AV*

loss��?)]�       ��(	mI�B=�AV*


acc*?�d�       ��2	�T�B=�AW*

val_loss�g?���6       `/�#	T�B=�AW*

val_acc{�?�hfi       ��-		T�B=�AW*

loss�a?DW�>       ��(	�	T�B=�AW*


acc�7?E���       ��2	�e\�B=�AX*

val_loss�m?J{��       `/�#	�f\�B=�AX*

val_acc�?z\r<       ��-	kg\�B=�AX*

loss�2?�L<�       ��(	�g\�B=�AX*


acc�?#�{)       ��2	Ne�B=�AY*

val_loss:?���       `/�#	e�B=�AY*

val_acc�?P��4       ��-	e�B=�AY*

loss3�?4F(�       ��(	�e�B=�AY*


acc+z?h�(#       ��2	Q�m�B=�AZ*

val_loss�r?/D�       `/�#	��m�B=�AZ*

val_acc|?�B�       ��-	�m�B=�AZ*

lossN.?=+J       ��(	{�m�B=�AZ*


acc��?��       ��2	Bx�B=�A[*

val_loss��?]� �       `/�#	��x�B=�A[*

val_acc��?�]�%       ��-	��x�B=�A[*

lossGB?�U�       ��(	=�x�B=�A[*


acc�G?>���       ��2	���B=�A\*

val_loss`?^^y�       `/�#	���B=�A\*

val_acc�?�u�       ��-	��B=�A\*

loss�??���        ��(	���B=�A\*


acc�?ċ�       ��2	�Չ�B=�A]*

val_loss��?VzD`       `/�#	�։�B=�A]*

val_acc�?䈻G       ��-	׉�B=�A]*

loss�+?7��D       ��(	k׉�B=�A]*


acc��?�d
       ��2	����B=�A^*

val_lossd4?�ؽ�       `/�#	����B=�A^*

val_acc?</w       ��-	`���B=�A^*

loss��?�L�t       ��(	����B=�A^*


accA�?4��       ��2	���B=�A_*

val_loss�?��       `/�#	���B=�A_*

val_acc��?cJ)�       ��-	G��B=�A_*

lossD�?f�x�       ��(	���B=�A_*


acc�??���       ��2	�Ȣ�B=�A`*

val_lossO�?��'       `/�#	-ʢ�B=�A`*

val_acc��?����       ��-	�ʢ�B=�A`*

loss+�?~?�d       ��(	!ˢ�B=�A`*


acc߁?���       ��2	�*��B=�Aa*

val_loss��?P���       `/�#	�+��B=�Aa*

val_acc�?��$       ��-	�,��B=�Aa*

loss��?B�       ��(	M-��B=�Aa*


accX?�Q �       ��2	�a��B=�Ab*

val_lossY�?�ٽ�       `/�#	sc��B=�Ab*

val_acc�?���       ��-	0d��B=�Ab*

lossľ?��       ��(	�d��B=�Ab*


acc��?��X�       ��2	����B=�Ac*

val_loss�&?;�       `/�#	ȵ��B=�Ac*

val_acc	?�g�       ��-	B���B=�Ac*

loss��?�}��       ��(	����B=�Ac*


acc�^?�(�