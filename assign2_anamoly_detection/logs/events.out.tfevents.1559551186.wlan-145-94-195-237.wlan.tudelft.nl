       �K"	  �47=�Abrain.Event:2�
���     ��$	Rc�47=�A"��
j
input_1Placeholder*
dtype0*'
_output_shapes
:���������+*
shape:���������+
m
dense_1/random_uniform/shapeConst*
valueB"+      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *dF��*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *dF�>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:+*
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

:+
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+
�
dense_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:+*
	container *
shape
:+
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+
Z
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
validate_shape(*
_output_shapes
:*
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
:
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������
g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������
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
:���������
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
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *   �*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2�ߜ*

seed
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes

:*
T0
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:
�
dense_2/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:
Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_2/ReluReludense_2/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
_
dense_3/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *q�?*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*

seed*
T0*
dtype0*
_output_shapes

:*
seed2��
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:
�
dense_3/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:
Z
dense_3/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_3/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_4/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"   +   
_
dense_4/random_uniform/minConst*
valueB
 *S���*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *S��>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
_output_shapes

:+*
seed2�Ֆ*

seed
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:+
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:+*
T0
�
dense_4/kernel
VariableV2*
_output_shapes

:+*
	container *
shape
:+*
shared_name *
dtype0
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*!
_class
loc:@dense_4/kernel
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:+
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
T0*'
_output_shapes
:���������+*
transpose_a( *
transpose_b( 
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*'
_output_shapes
:���������+*
T0*
data_formatNHWC
W
dense_4/ReluReludense_4/BiasAdd*'
_output_shapes
:���������+*
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
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
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*
	keep_dims( *

Tidx0*
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
:���������*
	keep_dims( *

Tidx0
|
loss/dense_4_loss/mulMulloss/dense_4_loss/Mean_1dense_4_sample_weights*#
_output_shapes
:���������*
T0
a
loss/dense_4_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
loss/dense_4_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/Castloss/dense_4_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
: *
	keep_dims( *

Tidx0*
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
metrics/acc/ArgMax_1/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
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
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape*+
_class!
loc:@loss/dense_4_loss/Mean_3*#
_output_shapes
:���������*

Tmultiples0*
T0
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
:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_3*
dtype0*
_output_shapes
:
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
>training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Maximum*+
_class!
loc:@loss/dense_4_loss/Mean_3*
_output_shapes
: *
T0
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
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*'
_output_shapes
:���������*

Tmultiples0
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*
out_type0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
valueB *,
_class"
 loc:@loss/dense_4_loss/truediv*
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
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv
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
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
�
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv
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
:���������
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
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������
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
:���������
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
6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*
T0*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������
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
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Mulloss/dense_4_loss/Mean_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*(
_class
loc:@loss/dense_4_loss/mul*#
_output_shapes
:���������*
T0
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
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*#
_output_shapes
:���������*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������
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
9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/modFloorMod9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/add:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *+
_class!
loc:@loss/dense_4_loss/Mean_1
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/deltaConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
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
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitchDynamicStitch;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range9training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/mod;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Fill*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
N
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1*
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeCtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/DynamicStitch*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������*
T0
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
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
dtype0*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1Maximum<training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod_1Atraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1/y*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
@training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/floordiv_1FloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Prod?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum_1*+
_class!
loc:@loss/dense_4_loss/Mean_1*
_output_shapes
: *
T0
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
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean
�
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0
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
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
T0*

index_type0*)
_class
loc:@loss/dense_4_loss/Mean*
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
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean
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
T0*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
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
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0
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
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
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
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
_output_shapes
:+*
T0*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC
�
2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Relu_grad/ReluGraddense_4/kernel/read*
T0*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh2training/Adam/gradients/dense_4/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_class
loc:@dense_4/MatMul*
_output_shapes

:+*
transpose_a(
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*'
_output_shapes
:���������*
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
:
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:���������*
transpose_a( 
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_2/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
N*'
_output_shapes
:���������*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
_output_shapes
:*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC
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
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_1/MatMul*
_output_shapes

:+*
transpose_a(*
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
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
h
training/Adam/zerosConst*
dtype0*
_output_shapes

:+*
valueB+*    
�
training/Adam/Variable
VariableV2*
dtype0*
_output_shapes

:+*
	container *
shape
:+*
shared_name 
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes

:+*
T0
b
training/Adam/zeros_1Const*
_output_shapes
:*
valueB*    *
dtype0
�
training/Adam/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
validate_shape(*
_output_shapes
:*
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
:
j
training/Adam/zeros_2Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_2
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:
b
training/Adam/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_3
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:
j
training/Adam/zeros_4Const*
dtype0*
_output_shapes

:*
valueB*    
�
training/Adam/Variable_4
VariableV2*
_output_shapes

:*
	container *
shape
:*
shared_name *
dtype0
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
_output_shapes

:*
T0*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
T0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
:
j
training/Adam/zeros_6Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*
_output_shapes

:+*
	container *
shape
:+
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:+
b
training/Adam/zeros_7Const*
_output_shapes
:+*
valueB+*    *
dtype0
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
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:+
j
training/Adam/zeros_8Const*
_output_shapes

:+*
valueB+*    *
dtype0
�
training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
_output_shapes

:+*
	container *
shape
:+
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+
b
training/Adam/zeros_9Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:
k
training/Adam/zeros_10Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_10
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:
c
training/Adam/zeros_11Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_11
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
k
training/Adam/zeros_12Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_12
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_13
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:
k
training/Adam/zeros_14Const*
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_14
VariableV2*
dtype0*
_output_shapes

:+*
	container *
shape
:+*
shared_name 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
_output_shapes

:+*
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
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
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
_output_shapes
:
p
&training/Adam/zeros_19/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
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
dtype0*
_output_shapes
:*
	container *
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
training/Adam/zeros_22Fill&training/Adam/zeros_22/shape_as_tensortraining/Adam/zeros_22/Const*

index_type0*
_output_shapes
:*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:*
T0
r
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
_output_shapes

:+*
T0
Z
training/Adam/sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
�
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:+
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes

:+*
T0
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:+
Z
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
}
training/Adam/SquareSquare4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:+*
T0
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes

:+*
T0
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes

:+
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
_output_shapes

:+*
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
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
_output_shapes

:+*
T0
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
_output_shapes

:+*
T0
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:+
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

:+*
T0
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes

:+*
T0
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:+
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable
�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
validate_shape(*
_output_shapes

:+*
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

:+
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes
:*
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
training/Adam/mul_7Multraining/Adam/sub_58training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes
:*
T0
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
training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
:
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
:
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
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
_output_shapes
:*
T0
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
:*
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
T0*
_output_shapes
:
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
:
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes
:
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
_output_shapes
:*
T0
�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:*
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
:
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
_output_shapes

:*
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

:*
T0
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:
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

:
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*
_output_shapes

:
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0*
_output_shapes

:
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

:
�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*
_output_shapes

:
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:
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

:
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
_output_shapes

:*
T0
�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
_output_shapes
:*
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
:
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
_output_shapes
:*
T0
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
training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:*
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:
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
:
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
_output_shapes
:*
T0
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:
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
:
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
:
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*
_output_shapes

:
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
training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*
_output_shapes

:
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
_output_shapes

:*
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
training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*
_output_shapes

:
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
_output_shapes

:*
T0
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*
_output_shapes

:
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

:
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
_output_shapes

:*
T0
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

:
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
_output_shapes

:*
T0
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes
:*
T0
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
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes
:*
T0
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:
[
training/Adam/sub_18/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes
:*
T0
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:*
T0
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
_output_shapes
: *
valueB
 *  �*
dtype0
�
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
_output_shapes
:*
T0
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
:
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:
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
:*
T0
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
:*
T0
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:*
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
:
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:+
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
training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:+*
T0
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:+
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:+
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

:+
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes

:+*
T0
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:+
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
_output_shapes

:+*
T0
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

:+
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:+
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

:+
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:+
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
_output_shapes

:+*
T0
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:+
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
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializedAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
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
IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_6
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
IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_11*
dtype0
�
IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_12*
dtype0
�
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13
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
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^dense_4/bias/Assign^dense_4/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign"CG��*�     �:S[	V��47=�AJ��
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
valueB"+      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *dF��*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *dF�>*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*

seed*
T0*
dtype0*
seed2끞*
_output_shapes

:+
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes

:+*
T0
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:+
�
dense_1/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:+*
shape
:+*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+*
use_locking(
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:+
Z
dense_1/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
x
dense_1/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
_output_shapes
:*
T0*
_class
loc:@dense_1/bias
�
dense_1/MatMulMatMulinput_1dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_1/TanhTanhdense_1/BiasAdd*'
_output_shapes
:���������*
T0
g
 dense_1/activity_regularizer/AbsAbsdense_1/Tanh*
T0*'
_output_shapes
:���������
g
"dense_1/activity_regularizer/mul/xConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
 dense_1/activity_regularizer/mulMul"dense_1/activity_regularizer/mul/x dense_1/activity_regularizer/Abs*'
_output_shapes
:���������*
T0
s
"dense_1/activity_regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
 dense_1/activity_regularizer/SumSum dense_1/activity_regularizer/mul"dense_1/activity_regularizer/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *   �*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *   ?*
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

:
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes

:*
T0
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes

:
�
dense_2/kernel
VariableV2*
dtype0*
	container *
_output_shapes

:*
shape
:*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
{
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes

:*
T0*!
_class
loc:@dense_2/kernel
Z
dense_2/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
x
dense_2/bias
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
�
dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
W
dense_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_3/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
valueB
 *q��*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *q�?*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*

seed*
T0*
dtype0*
seed2��*
_output_shapes

:
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:*
T0
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:*
T0
�
dense_3/kernel
VariableV2*
shape
:*
shared_name *
dtype0*
	container *
_output_shapes

:
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
{
dense_3/kernel/readIdentitydense_3/kernel*
_output_shapes

:*
T0*!
_class
loc:@dense_3/kernel
Z
dense_3/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
x
dense_3/bias
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias
�
dense_3/MatMulMatMuldense_2/Reludense_3/kernel/read*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
W
dense_3/TanhTanhdense_3/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_4/random_uniform/shapeConst*
valueB"   +   *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *S���*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
_output_shapes
: *
valueB
 *S��>*
dtype0
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seed2�Ֆ*
_output_shapes

:+*

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

:+
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:+
�
dense_4/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*!
_class
loc:@dense_4/kernel
{
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel*
_output_shapes

:+*
T0
Z
dense_4/ConstConst*
valueB+*    *
dtype0*
_output_shapes
:+
x
dense_4/bias
VariableV2*
dtype0*
	container *
_output_shapes
:+*
shape:+*
shared_name 
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*
_class
loc:@dense_4/bias
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
dense_4/ReluReludense_4/BiasAdd*'
_output_shapes
:���������+*
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
	container *
_output_shapes
: *
shape: 
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
dtype0*
	container *
_output_shapes
: *
shape: 
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
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
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
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
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
Adam/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
n

Adam/decay
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
: *
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
dense_4_sample_weightsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
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
(loss/dense_4_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
loss/dense_4_loss/MeanMeanloss/dense_4_loss/Square(loss/dense_4_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
m
*loss/dense_4_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Mean*loss/dense_4_loss/Mean_1/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
loss/dense_4_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss/dense_4_loss/Mean_3Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_1*

Tidx0*
	keep_dims( *
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
metrics/acc/ArgMax_1ArgMaxdense_4/Relumetrics/acc/ArgMax_1/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
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
metrics/acc/MeanMeanmetrics/acc/Castmetrics/acc/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/add*
valueB *
dtype0*
_output_shapes
: 
�
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
_class
loc:@loss/add*
valueB
 *  �?
�
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_class
loc:@loss/add*

index_type0*
_output_shapes
: *
T0
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
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shapeConst*+
_class!
loc:@loss/dense_4_loss/Mean_3*
valueB:*
dtype0*
_output_shapes
:
�
=training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_3_grad/Reshape/shape*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
Tshape0
�
;training/Adam/gradients/loss/dense_4_loss/Mean_3_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_3*
out_type0*
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
Ktraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
valueB"      
�
Etraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeReshapetraining/Adam/gradients/FillKtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Reshape/shape*
_output_shapes

:*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
Tshape0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ShapeShape dense_1/activity_regularizer/mul*
_output_shapes
:*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*
out_type0
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/TileTileEtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Shape*

Tmultiples0*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Sum*'
_output_shapes
:���������
�
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*,
_class"
 loc:@loss/dense_4_loss/truediv*
out_type0*
_output_shapes
:*
T0
�
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *,
_class"
 loc:@loss/dense_4_loss/truediv*
valueB 
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
:*
	keep_dims( *

Tidx0
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
:���������
�
Atraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumSumAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/MulStraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/ReshapeReshapeAtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape*
_output_shapes
: *
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Mul"dense_1/activity_regularizer/mul/xBtraining/Adam/gradients/dense_1/activity_regularizer/Sum_grad/Tile*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*'
_output_shapes
:���������
�
Ctraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1SumCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Mul_1Utraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/BroadcastGradientArgs:1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Gtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1ReshapeCtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Sum_1Etraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Shape_1*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/mul*
Tshape0*'
_output_shapes
:���������
�
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean_1*
_output_shapes
:*
T0*(
_class
loc:@loss/dense_4_loss/mul*
out_type0
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
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/MulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*
T0*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*#
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
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
�
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*#
_output_shapes
:���������*
T0*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0
�
Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/SignSigndense_1/Tanh*'
_output_shapes
:���������*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs
�
Atraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mulMulGtraining/Adam/gradients/dense_1/activity_regularizer/mul_grad/Reshape_1Btraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/Sign*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*'
_output_shapes
:���������
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
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/rangeRangeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/start:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/SizeAtraining/Adam/gradients/loss/dense_4_loss/Mean_1_grad/range/delta*
_output_shapes
:*

Tidx0*+
_class!
loc:@loss/dense_4_loss/Mean_1
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
?training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_1*
value	B :
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
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_3Shapeloss/dense_4_loss/Mean_1*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
out_type0*
_output_shapes
:
�
;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ConstConst*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Shape_2;training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_1
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Const_1Const*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_1*
valueB: *
dtype0
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

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_1*
Truncate( *

DstT0*
_output_shapes
: 
�
=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/Cast*+
_class!
loc:@loss/dense_4_loss/Mean_1*#
_output_shapes
:���������*
T0
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
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
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
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape=training/Adam/gradients/loss/dense_4_loss/Mean_1_grad/truedivAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
Tshape0
�
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*0
_output_shapes
:������������������
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
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*)
_class
loc:@loss/dense_4_loss/Mean
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
6training/Adam/gradients/loss/dense_4_loss/sub_grad/SumSum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Htraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/sub
�
:training/Adam/gradients/loss/dense_4_loss/sub_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum8training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape*
T0*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*'
_output_shapes
:���������+
�
8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1Sum;training/Adam/gradients/loss/dense_4_loss/Square_grad/Mul_1Jtraining/Adam/gradients/loss/dense_4_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:
�
6training/Adam/gradients/loss/dense_4_loss/sub_grad/NegNeg8training/Adam/gradients/loss/dense_4_loss/sub_grad/Sum_1*(
_class
loc:@loss/dense_4_loss/sub*
_output_shapes
:*
T0
�
<training/Adam/gradients/loss/dense_4_loss/sub_grad/Reshape_1Reshape6training/Adam/gradients/loss/dense_4_loss/sub_grad/Neg:training/Adam/gradients/loss/dense_4_loss/sub_grad/Shape_1*(
_class
loc:@loss/dense_4_loss/sub*
Tshape0*0
_output_shapes
:������������������*
T0
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
transpose_a( *'
_output_shapes
:���������*
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

:+*
transpose_b( 
�
2training/Adam/gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh2training/Adam/gradients/dense_4/MatMul_grad/MatMul*
T0*
_class
loc:@dense_3/Tanh*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(*
_output_shapes

:*
transpose_b( 
�
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*
_class
loc:@dense_2/Relu*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_2/kernel/read*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(*
_output_shapes

:*
transpose_b( 
�
training/Adam/gradients/AddNAddNAtraining/Adam/gradients/dense_1/activity_regularizer/Abs_grad/mul2training/Adam/gradients/dense_2/MatMul_grad/MatMul*
T0*3
_class)
'%loc:@dense_1/activity_regularizer/Abs*
N*'
_output_shapes
:���������
�
2training/Adam/gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhtraining/Adam/gradients/AddN*
T0*
_class
loc:@dense_1/Tanh*'
_output_shapes
:���������
�
8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
2training/Adam/gradients/dense_1/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*!
_class
loc:@dense_1/MatMul*
transpose_a( *'
_output_shapes
:���������+*
transpose_b(*
T0
�
4training/Adam/gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_12training/Adam/gradients/dense_1/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_1/MatMul*
transpose_a(*
_output_shapes

:+*
transpose_b( 
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
�
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0	*"
_class
loc:@Adam/iterations
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
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable
VariableV2*
dtype0*
	container *
_output_shapes

:+*
shape
:+*
shared_name 
�
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
_output_shapes

:+*
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

:+
b
training/Adam/zeros_1Const*
dtype0*
_output_shapes
:*
valueB*    
�
training/Adam/Variable_1
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
:
j
training/Adam/zeros_2Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_2
VariableV2*
shape
:*
shared_name *
dtype0*
	container *
_output_shapes

:
�
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes

:
b
training/Adam/zeros_3Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_3
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
:
j
training/Adam/zeros_4Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_4
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:*
shape
:
�
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
_output_shapes

:*
T0*+
_class!
loc:@training/Adam/Variable_4
b
training/Adam/zeros_5Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_5
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_5
j
training/Adam/zeros_6Const*
dtype0*
_output_shapes

:+*
valueB+*    
�
training/Adam/Variable_6
VariableV2*
dtype0*
	container *
_output_shapes

:+*
shape
:+*
shared_name 
�
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes

:+
b
training/Adam/zeros_7Const*
dtype0*
_output_shapes
:+*
valueB+*    
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
valueB+*    *
dtype0*
_output_shapes

:+
�
training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:+*
shape
:+
�
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes

:+*
T0
b
training/Adam/zeros_9Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_9
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(
�
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
:
k
training/Adam/zeros_10Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_10
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:*
shape
:
�
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes

:
c
training/Adam/zeros_11Const*
dtype0*
_output_shapes
:*
valueB*    
�
training/Adam/Variable_11
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
�
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:
k
training/Adam/zeros_12Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/Adam/Variable_12
VariableV2*
	container *
_output_shapes

:*
shape
:*
shared_name *
dtype0
�
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*
_output_shapes

:
c
training/Adam/zeros_13Const*
valueB*    *
dtype0*
_output_shapes
:
�
training/Adam/Variable_13
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
:
k
training/Adam/zeros_14Const*
_output_shapes

:+*
valueB+*    *
dtype0
�
training/Adam/Variable_14
VariableV2*
dtype0*
	container *
_output_shapes

:+*
shape
:+*
shared_name 
�
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+*
use_locking(
�
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes

:+
c
training/Adam/zeros_15Const*
dtype0*
_output_shapes
:+*
valueB+*    
�
training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:+*
shape:+
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
training/Adam/zeros_17Fill&training/Adam/zeros_17/shape_as_tensortraining/Adam/zeros_17/Const*

index_type0*
_output_shapes
:*
T0
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
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
:*
use_locking(
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
training/Adam/zeros_18/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*
_output_shapes
:
�
training/Adam/Variable_18
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
�
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*
_output_shapes
:*
use_locking(
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
training/Adam/zeros_19Fill&training/Adam/zeros_19/shape_as_tensortraining/Adam/zeros_19/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_19
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19
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
training/Adam/zeros_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
training/Adam/zeros_20Fill&training/Adam/zeros_20/shape_as_tensortraining/Adam/zeros_20/Const*
_output_shapes
:*
T0*

index_type0
�
training/Adam/Variable_20
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
�
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(
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
training/Adam/zeros_22/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*
_output_shapes
:
�
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_22
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
training/Adam/zeros_23Fill&training/Adam/zeros_23/shape_as_tensortraining/Adam/zeros_23/Const*

index_type0*
_output_shapes
:*
T0
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

:+
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
training/Adam/mul_2Multraining/Adam/sub_24training/Adam/gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:+*
T0
m
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes

:+
t
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes

:+
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

:+
n
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes

:+
m
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*
_output_shapes

:+
k
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*
_output_shapes

:+
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *  �
�
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*
_output_shapes

:+
�
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes

:+
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

:+
u
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes

:+
q
training/Adam/sub_4Subdense_1/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes

:+
�
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
�
training/Adam/Assign_2Assigndense_1/kerneltraining/Adam/sub_4*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
:
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
:
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
:
p
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes
:
Z
training/Adam/sub_6/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_1Square8training/Adam/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes
:*
T0
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes
:*
T0
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes
:*
T0
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
_output_shapes
: *
valueB
 *  �*
dtype0
�
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
_output_shapes
:*
T0
�
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
T0*
_output_shapes
:
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
:*
T0
Z
training/Adam/add_6/yConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes
:*
T0
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
:*
T0
k
training/Adam/sub_7Subdense_1/bias/readtraining/Adam/truediv_2*
_output_shapes
:*
T0
�
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(
�
training/Adam/Assign_5Assigndense_1/biastraining/Adam/sub_7*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*
_output_shapes

:
Z
training/Adam/sub_8/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
o
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*
_output_shapes

:
v
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0*
_output_shapes

:
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

:
q
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*
_output_shapes

:
o
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
_output_shapes

:*
T0
l
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
_output_shapes

:*
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
_output_shapes
: *
valueB
 *  �*
dtype0
�
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*
_output_shapes

:*
T0
�
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
_output_shapes

:*
T0
d
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*
_output_shapes

:
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

:
v
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0*
_output_shapes

:
r
training/Adam/sub_10Subdense_2/kernel/readtraining/Adam/truediv_3*
T0*
_output_shapes

:
�
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
training/Adam/Assign_8Assigndense_2/kerneltraining/Adam/sub_10*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
:
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
:
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes
:
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
_output_shapes
:*
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
:
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
:*
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
:
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes
:
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
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
_output_shapes
:*
T0
�
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
:
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes
:
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
:
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes
:*
T0
l
training/Adam/sub_13Subdense_2/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
:
�
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3
�
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_11Assigndense_2/biastraining/Adam/sub_13*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
u
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
_output_shapes

:*
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

:
p
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
_output_shapes

:*
T0
v
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0*
_output_shapes

:
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

:
r
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
_output_shapes

:*
T0
p
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
_output_shapes

:*
T0
m
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
_output_shapes

:*
T0
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
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
_output_shapes

:*
T0
�
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*
_output_shapes

:
d
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
_output_shapes

:*
T0
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

:*
T0
w
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*
_output_shapes

:
r
training/Adam/sub_16Subdense_3/kernel/readtraining/Adam/truediv_5*
T0*
_output_shapes

:
�
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
_output_shapes

:
�
training/Adam/Assign_14Assigndense_3/kerneltraining/Adam/sub_16*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes
:*
T0
[
training/Adam/sub_17/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
_output_shapes
: *
T0
�
training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:
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
:
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:*
T0
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:
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
:
�
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
_output_shapes
:*
T0
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
_output_shapes
:*
T0
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
:
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
:*
T0
l
training/Adam/sub_19Subdense_3/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:
�
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:
�
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/Adam/Assign_17Assigndense_3/biastraining/Adam/sub_19*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:
u
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes

:+
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

:+
p
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes

:+
v
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes

:+
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

:+
r
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes

:+
p
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes

:+
m
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes

:+
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
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
_output_shapes

:+*
T0
�
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*
T0*
_output_shapes

:+
d
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
T0*
_output_shapes

:+
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

:+*
T0
w
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes

:+
r
training/Adam/sub_22Subdense_4/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes

:+
�
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*
_output_shapes

:+
�
training/Adam/Assign_20Assigndense_4/kerneltraining/Adam/sub_22*
validate_shape(*
_output_shapes

:+*
use_locking(*
T0*!
_class
loc:@dense_4/kernel
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
training/Adam/sub_24/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
_output_shapes
: *
T0
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
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:+*
use_locking(
�
training/Adam/Assign_23Assigndense_4/biastraining/Adam/sub_25*
validate_shape(*
_output_shapes
:+*
use_locking(*
T0*
_class
loc:@dense_4/bias
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
dtype0*
_output_shapes
: *
_class
loc:@dense_2/bias
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
IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_5
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
IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_9*
dtype0
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
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable_16*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_16
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
IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_22*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_22*
dtype0
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
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08����       ���	���47=�A*

val_loss��?�4�       �	���47=�A*

val_acc�_�=/�       �K"	n��47=�A*

loss���?>ؼ       ���	���47=�A*


acc�;=�X�       ��2	Ϧ�47=�A*

val_loss���?`��v       `/�#	q��47=�A*

val_acc�f�=ի��       ��-	.��47=�A*

lossg޸?%͖�       ��(	Ʃ�47=�A*


acc5ٍ=w�       ��2	Tr�47=�A*

val_loss�<z?�6�       `/�#	�t�47=�A*

val_acc�v�=�8       ��-	�u�47=�A*

loss${�?2y��       ��(	5v�47=�A*


acc�"�=8�G       ��2	���47=�A*

val_loss[O\?I[8       `/�#	��47=�A*

val_accq2�=�8*�       ��-	ƨ�47=�A*

loss�]q?Gx��       ��(	e��47=�A*


acc��j=I��|       ��2	2u57=�A*

val_loss0�N?,��D       `/�#	�v57=�A*

val_acc���=o��       ��-	hw57=�A*

lossU]?ju�d       ��(	�w57=�A*


acc��e=�5>M       ��2	��57=�A*

val_loss�`F?�3       `/�#	��57=�A*

val_acc+�>�Cg�       ��-	H�57=�A*

loss�R?��'       ��(	��57=�A*


acc�+>���m       ��2	�57=�A*

val_loss@?o��       `/�#	�57=�A*

val_acc��5>Y�l       ��-	Z57=�A*

lossbTJ?"���       ��(	�57=�A*


accG�1>=^�       ��2	�$57=�A*

val_loss�d;?�H       `/�#	h $57=�A*

val_accWQ>�r�       ��-	%!$57=�A*

loss��D?T��       ��(	�!$57=�A*


acc~�O>1B�       ��2	��/57=�A*

val_loss��7?�
V�       `/�#	]�/57=�A*

val_acc�\>Ǳ�p       ��-	�/57=�A*

loss&K@?�]p�       ��(	��/57=�A*


accv#h>M�o(       ��2	�a:57=�A	*

val_loss#�5?��       `/�#	�b:57=�A	*

val_acc��^>�0��       ��-	�b:57=�A	*

loss$=?ƍ#�       ��(	Nc:57=�A	*


acc�o>�$�*       ��2		E57=�A
*

val_loss�g3?�LF       `/�#	E57=�A
*

val_accy`>5vF�       ��-	�E57=�A
*

loss�{:?y��T       ��(	(E57=�A
*


acc��m>����       ��2	0P57=�A*

val_loss��1?n	 �       `/�#	�P57=�A*

val_acc��_>ܽW�       ��-	]P57=�A*

lossPU8?��e�       ��(	�P57=�A*


acc�,n>�'�Z       ��2	7�Z57=�A*

val_loss��0?�ⷚ       `/�#	.�Z57=�A*

val_acc��T>}ֈ       ��-	��Z57=�A*

lossr�6?����       ��(	��Z57=�A*


acc�j>�T       ��2	e�f57=�A*

val_loss�{/?m0l&       `/�#	�f57=�A*

val_accÏj>xU�       ��-	��f57=�A*

loss,5?<K��       ��(	e�f57=�A*


acc�g>�X"       ��2	A�r57=�A*

val_loss_Y.?�p��       `/�#	�r57=�A*

val_acc�MM>3U�       ��-	��r57=�A*

loss�4?3���       ��(	<�r57=�A*


acc��p>7^q       ��2	��}57=�A*

val_lossU�-?8��       `/�#	 �}57=�A*

val_acck�R>(� e       ��-	��}57=�A*

loss��2?���       ��(	]�}57=�A*


acc�no>�[��       ��2	g�57=�A*

val_lossT�,?���V       `/�#	��57=�A*

val_acc�]T>D��h       ��-	��57=�A*

lossR�1?E�8L       ��(	@�57=�A*


accxhk>,t��       ��2	K��57=�A*

val_lossI�+?�S�       `/�#	獔57=�A*

val_acc��[>�J�       ��-	���57=�A*

loss��0?#���       ��(	C��57=�A*


acc�t>U��       ��2	ˇ�57=�A*

val_loss�2+?�       `/�#	���57=�A*

val_acc<~>�C��       ��-	ڊ�57=�A*

loss4�/?'�       ��(	��57=�A*


acc$Vt>P�-�       ��2	N��57=�A*

val_loss�*?���l       `/�#	Ĵ�57=�A*

val_acc�v_>�z       ��-	���57=�A*

loss'/?L��g       ��(	=��57=�A*


acc��v>���       ��2	�Y�57=�A*

val_loss�)?���       `/�#	�[�57=�A*

val_acc	�k>��O       ��-	J\�57=�A*

loss��-?�|~�       ��(	�\�57=�A*


accY {>�cBv       ��2	��57=�A*

val_loss�)?ac �       `/�#	���57=�A*

val_acc��P>�7�       ��-	U��57=�A*

loss1�,?�J�%       ��(	��57=�A*


acc�>U��L       ��2	Z��57=�A*

val_loss��(?'���       `/�#	f��57=�A*

val_accL>>���       ��-	��57=�A*

lossgN,?�X��       ��(	L��57=�A*


acc2�}>�!�       ��2	�/�57=�A*

val_lossk(?�Ď�       `/�#	�1�57=�A*

val_acc	�k>&w��       ��-	^2�57=�A*

loss�_+?rf��       ��(	�2�57=�A*


acc�E�>�K       ��2	[�57=�A*

val_loss/�'?a|�       `/�#	��57=�A*

val_acc��^>Q�ss       ��-	��57=�A*

loss��*?m�NL       ��(	N�57=�A*


acc5\�>T�IZ       ��2	w2�57=�A*

val_lossݪ&?˹=C       `/�#	4�57=�A*

val_acc}7|>��       ��-	�4�57=�A*

loss�0*?s�K�       ��(	n5�57=�A*


acct�>�e��       ��2	���57=�A*

val_loss�&?W�       `/�#	��57=�A*

val_acc��>&�       ��-	ݹ�57=�A*

lossQ\)?<�w       ��(	|��57=�A*


acc��>W�       ��2	�67=�A*

val_loss�#%?"c�       `/�#	 67=�A*

val_acc�V�>=g       ��-	�67=�A*

loss�(?�fI�       ��(	|67=�A*


acc>8�>����       ��2	�C67=�A*

val_loss��$?O�֌       `/�#	-E67=�A*

val_accc�>��K       ��-	�E67=�A*

loss�U'?̘�u       ��(	�F67=�A*


accZғ>�VT�       ��2	 67=�A*

val_lossV)#?@S�c       `/�#	�67=�A*

val_accw�>���       ��-	167=�A*

loss$&?�GLs       ��(	�67=�A*


acc��>��z�       ��2	@�%67=�A*

val_loss�2#?��'�       `/�#	��%67=�A*

val_acc���>�W�       ��-	P�%67=�A*

loss�7%?rĞ�       ��(	��%67=�A*


acc�>!
MU       ��2	]�067=�A*

val_loss�*"?y��U       `/�#	��067=�A*

val_accT�>�q#       ��-	��067=�A*

loss�$?�ZRt       ��(	P�067=�A*


accy��>�O�f       ��2	\�=67=�A *

val_loss��!?�	hj       `/�#	��=67=�A *

val_acc�ט>�IH
       ��-	�=67=�A *

lossO�#?|4�A       ��(	q�=67=�A *


acc	ǫ>pA��       ��2	�H67=�A!*

val_loss� ?�N       `/�#	�H67=�A!*

val_acc�Ͱ>�!       ��-	�H67=�A!*

loss[#?^d�/       ��(	�H67=�A!*


acc8�>��U�       ��2	��R67=�A"*

val_loss<� ?���       `/�#	K�R67=�A"*

val_accYҲ>�+Z       ��-	�R67=�A"*

lossP�"?���s       ��(	��R67=�A"*


acc��>�f�       ��2	�]67=�A#*

val_lossD ?0��       `/�#	��]67=�A#*

val_accn۶>(su�       ��-	<�]67=�A#*

loss-,"?��^       ��(	܂]67=�A#*


acc�y�>�7�<       ��2	uzh67=�A$*

val_loss  ?h��Z       `/�#	�{h67=�A$*

val_acc&��>��,c       ��-	�|h67=�A$*

lossį!?���A       ��(	1}h67=�A$*


acc��>;8�       ��2	��r67=�A%*

val_losseX?趁�       `/�#	z�r67=�A%*

val_accW��>���       ��-	S�r67=�A%*

loss&X!?�R��       ��(	��r67=�A%*


acc���>�ဒ       ��2	b�}67=�A&*

val_loss^5?jl-{       `/�#	��}67=�A&*

val_acc<��>z��       ��-	��}67=�A&*

loss� ?�d$<       ��(	U�}67=�A&*


accl�>#xv�       ��2	�o�67=�A'*

val_loss'?9��       `/�#	Yq�67=�A'*

val_acc;��>a�o�       ��-	r�67=�A'*

losss� ?1(�       ��(	�r�67=�A'*


acc��>I� �       ��2	%�67=�A(*

val_loss��?��       `/�#	��67=�A(*

val_acc���>���       ��-	y�67=�A(*

losss@ ?����       ��(	�67=�A(*


acc
E�>q�pB       ��2	P��67=�A)*

val_lossY�?-�       `/�#	.�67=�A)*

val_acc���>x�<>       ��-	��67=�A)*

loss-�?���       ��(	��67=�A)*


acc���>�5�        ��2	A/�67=�A**

val_lossa�?d��       `/�#	0�67=�A**

val_acc��>&�*        ��-	�0�67=�A**

loss� ?�U�       ��(	�0�67=�A**


acc"U�>Y��       ��2	�S�67=�A+*

val_loss��?=[ڸ       `/�#	�T�67=�A+*

val_acc���>�z��       ��-	U�67=�A+*

loss/]?���       ��(	uU�67=�A+*


acc���>S
Xl       ��2	��67=�A,*

val_loss�?�p��       `/�#	1��67=�A,*

val_acc�ݷ>�̌       ��-	���67=�A,*

loss�0?��9~       ��(	��67=�A,*


acc}{�>5 �       ��2	���67=�A-*

val_loss�@?��       `/�#	R��67=�A-*

val_accd��>��G       ��-	��67=�A-*

loss^�?;-R       ��(	���67=�A-*


acc��>E�s
       ��2	h]�67=�A.*

val_loss��??���       `/�#	1_�67=�A.*

val_acc�A�>���       ��-	�_�67=�A.*

loss#?�)��       ��(	�`�67=�A.*


acc���>>Hc�       ��2	��67=�A/*

val_loss��?Z{rX       `/�#	5��67=�A/*

val_acc<��>O3�       ��-	Ք�67=�A/*

loss�?Ai�>       ��(	`��67=�A/*


acc�5�>H�ї       ��2	\q�67=�A0*

val_lossL?7̒K       `/�#	Hr�67=�A0*

val_acc&��>�K       ��-	�r�67=�A0*

lossA�?р2I       ��(	s�67=�A0*


acc6��>��Q&       ��2	�$�67=�A1*

val_loss9�?Ow�       `/�#	�%�67=�A1*

val_acc��>]�[�       ��-	�'�67=�A1*

loss�m?/�-�       ��(	F(�67=�A1*


acc� �>&u�       ��2	�G�67=�A2*

val_loss�?�c�v       `/�#	�H�67=�A2*

val_acc���>
4;       ��-	QI�67=�A2*

loss ?$���       ��(	�I�67=�A2*


accw��>B���       ��2	���67=�A3*

val_loss�w?���1       `/�#	���67=�A3*

val_acc�A�>}5�       ��-	E��67=�A3*

loss �?��#       ��(	���67=�A3*


acc�d�>���       ��2	m77=�A4*

val_loss�4?�t       `/�#	�m77=�A4*

val_acc�T�>M���       ��-	en77=�A4*

loss#t?��|       ��(	�n77=�A4*


acc��>֧��       ��2	s*77=�A5*

val_lossG�?�$C       `/�#	�+77=�A5*

val_acc���>�O�       ��-	,77=�A5*

lossT?"�P       ��(	Z,77=�A5*


acc���>xy       ��2	:�77=�A6*

val_loss��?���       `/�#	��77=�A6*

val_acc���>"F�"       ��-	��77=�A6*

loss�?2g�       ��(	N�77=�A6*


acch��>#��       ��2	�m77=�A7*

val_loss_�?��s       `/�#	;o77=�A7*

val_acc�v�>F��       ��-	�o77=�A7*

lossn�?�r~�       ��(	�p77=�A7*


acc�g�>1�x�       ��2	�.'77=�A8*

val_loss
X?����       `/�#	�0'77=�A8*

val_acc'��>_���       ��-	�1'77=�A8*

loss��?���       ��(	�2'77=�A8*


acc�E�>t�@�       ��2	�177=�A9*

val_lossi�?Vn�l       `/�#	 �177=�A9*

val_acc��>���}       ��-	~�177=�A9*

loss�U?Cg<       ��(	�177=�A9*


accd��>��7       ��2	{<77=�A:*

val_loss�+?�iy       `/�#	<77=�A:*

val_acc���>3e��       ��-	�<77=�A:*

loss�?�M��       ��(	z<77=�A:*


accBy�>�/BB       ��2	��F77=�A;*

val_loss��?�
"       `/�#	��F77=�A;*

val_acc	=�>.�       ��-	q�F77=�A;*

loss�?��U       ��(	��F77=�A;*


acc���>6       ��2	ʦQ77=�A<*

val_loss,X?�       `/�#	D�Q77=�A<*

val_acc���>Fe��       ��-	��Q77=�A<*

loss�%?�1l�       ��(	��Q77=�A<*


accX��>��X�       ��2	-	\77=�A=*

val_loss�?�l'�       `/�#	�
\77=�A=*

val_accx��>����       ��-	_\77=�A=*

loss�?W�w�       ��(	�\77=�A=*


acc���>#�       ��2	�d77=�A>*

val_lossV?2��       `/�#	�d77=�A>*

val_accd��>��y       ��-	?d77=�A>*

loss��?���       ��(	�d77=�A>*


acc�P�>��r@       ��2	��l77=�A?*

val_loss�?Nr       `/�#	��l77=�A?*

val_acc���>)&щ       ��-	�l77=�A?*

loss�'?dոV       ��(	e�l77=�A?*


acc�.�>�)j�       ��2	�t77=�A@*

val_lossC�?��Ջ       `/�#	ؠt77=�A@*

val_acck/�>��       ��-	M�t77=�A@*

loss�?��Z       ��(	��t77=�A@*


acc���>v��F       ��2	hV77=�AA*

val_loss�?k>�       `/�#	�W77=�AA*

val_acc]��>�|�       ��-	GX77=�AA*

loss�?�� �       ��(	�X77=�AA*


acc��>���N       ��2	���77=�AB*

val_lossf?w��d       `/�#	n��77=�AB*

val_accÏ�>Y*�       ��-	���77=�AB*

lossU�?�Ղi       ��(	X��77=�AB*


accE�>�t��       ��2	f��77=�AC*

val_loss�N?ob$       `/�#	E��77=�AC*

val_acc���>���       ��-	���77=�AC*

loss0�?E2�T       ��(	��77=�AC*


acc���>	0��       ��2	un�77=�AD*

val_loss��?Ѐ�"       `/�#	p�77=�AD*

val_acck/�>���       ��-	�p�77=�AD*

loss�W?�*�       ��(	�p�77=�AD*


accce�>�׉       ��2	�M�77=�AE*

val_loss?�G�       `/�#	�N�77=�AE*

val_acc��>�,       ��-	+O�77=�AE*

loss)�?�2'       ��(	�O�77=�AE*


acc���>���       ��2	�77=�AF*

val_loss��?�̦p       `/�#	��77=�AF*

val_acc	=�>p��       ��-	l�77=�AF*

loss�?8�GF       ��(	�77=�AF*


accƥ�>Ip       ��2	g�77=�AG*

val_loss��?��Yb       `/�#	�h�77=�AG*

val_acc��>�q5       ��-	wi�77=�AG*

loss5�?��@       ��(	j�77=�AG*


accKU�>ͬ�       ��2	M�77=�AH*

val_lossU�?9���       `/�#	��77=�AH*

val_acc��>�d��       ��-	��77=�AH*

loss+�?�K/k       ��(	Q�77=�AH*


acc���>�&7T       ��2	��77=�AI*

val_loss�?�`�x       `/�#	�77=�AI*

val_acc��>Se�       ��-	��77=�AI*

loss�R?w&|�       ��(	��77=�AI*


acc�9�>z3       ��2	���77=�AJ*

val_loss�?�u`<       `/�#	���77=�AJ*

val_acc:6�>��I|       ��-	v��77=�AJ*

loss�\?���I       ��(	���77=�AJ*


acc�&�>�^��       ��2	K��77=�AK*

val_loss�A?`�T]       `/�#	&��77=�AK*

val_acc��>͈]U       ��-	���77=�AK*

loss�<?�?��       ��(	���77=�AK*


acc���>��B�       ��2	8�77=�AL*

val_loss��?,k�/       `/�#	#�77=�AL*

val_acc~��>Z��T       ��-	��77=�AL*

lossG�?�!b�       ��(	��77=�AL*


acc���>k�[       ��2	���77=�AM*

val_loss�^?Lp�A       `/�#	<��77=�AM*

val_acc~��>��8       ��-	���77=�AM*

lossxg?L��       ��(	���77=�AM*


acc��>��W       ��2	�C�77=�AN*

val_loss�??<�:       `/�#	0E�77=�AN*

val_acc@�>��       ��-	�E�77=�AN*

loss��?��a       ��(	�F�77=�AN*


accۨ�>>�F       ��2	�q87=�AO*

val_loss�^?
`4       `/�#	.s87=�AO*

val_accGX�>��!�       ��-	�s87=�AO*

loss��?=��	       ��(	�t87=�AO*


acc_X�>wB7�       ��2	�m87=�AP*

val_loss	?�=�P       `/�#	�n87=�AP*

val_acc���>�`2�       ��-	+o87=�AP*

loss}�?I�f       ��(	�o87=�AP*


acci��>a;�       ��2	;87=�AQ*

val_lossg+?d�)l       `/�#	�87=�AQ*

val_acc�L�>..x       ��-	�87=�AQ*

loss��?��       ��(	?87=�AQ*


acc��>���       ��2	Ӧ!87=�AR*

val_loss�?�28�       `/�#	q�!87=�AR*

val_acc� �>\ˈ       ��-	3�!87=�AR*

loss�<?�rg�       ��(	ѩ!87=�AR*


accn	�>�%ߎ       ��2	2,87=�AS*

val_loss{�?���       `/�#	�3,87=�AS*

val_acc�L�>�f��       ��-	~4,87=�AS*

loss��?�)�       ��(	5,87=�AS*


acc�,�>=;�        ��2	�-787=�AT*

val_loss�?���       `/�#	'/787=�AT*

val_accG�>-\�Y       ��-	�/787=�AT*

loss��?���       ��(	�0787=�AT*


acce��>C�       ��2	ݹA87=�AU*

val_loss7�?Ʃ��       `/�#	w�A87=�AU*

val_acc8�>��u       ��-	4�A87=�AU*

loss&?�q��       ��(	ϼA87=�AU*


acc�,�>�a       ��2	��I87=�AV*

val_loss��?���r       `/�#	��I87=�AV*

val_acc��>Q��P       ��-	0�I87=�AV*

loss6?��n�       ��(	��I87=�AV*


acct[�>��P       ��2	[DT87=�AW*

val_lossI|?��%       `/�#	�FT87=�AW*

val_accGX�>�!ċ       ��-	�GT87=�AW*

loss�?�T�        ��(	'HT87=�AW*


acc�^�>�lL�       ��2	G#\87=�AX*

val_loss|�?��       `/�#	�$\87=�AX*

val_acc~��>f�a       ��-	%\87=�AX*

loss�?�p�p       ��(	p%\87=�AX*


acc��>_Ğ>       ��2	E�c87=�AY*

val_loss͉?@�Hh       `/�#	3�c87=�AY*

val_acc~��>NԷ�       ��-	��c87=�AY*

loss��?:�w�       ��(		�c87=�AY*


accT1�>a�17       ��2	�n87=�AZ*

val_loss�?�;�@       `/�#	��n87=�AZ*

val_acc��>�͜�       ��-	w�n87=�AZ*

loss��?''m�       ��(	�n87=�AZ*


acc�k�>�xm       ��2	��v87=�A[*

val_loss�j?!`�#       `/�#	��v87=�A[*

val_acc2O�>/�f�       ��-	�v87=�A[*

lossK?d?�       ��(	i�v87=�A[*


acc�{�>�O`�       ��2	���87=�A\*

val_loss��?م       `/�#	T��87=�A\*

val_accp�>Nx��       ��-	��87=�A\*

loss?W��       ��(	���87=�A\*


acc��>��I�       ��2	JΉ87=�A]*

val_loss	?��wk       `/�#	$ω87=�A]*

val_acc��>�	�       ��-	�ω87=�A]*

lossP�?*ƥ       ��(	�ω87=�A]*


accc��>��!�       ��2	�D�87=�A^*

val_lossd�?Z��r       `/�#	bF�87=�A^*

val_accF�>�j(       ��-	$G�87=�A^*

lossή?f��L       ��(	�G�87=�A^*


acc-��>�Sy       ��2	[��87=�A_*

val_loss�?�F       `/�#	܂�87=�A_*

val_acc��>�:�       ��-	Q��87=�A_*

loss1�?���       ��(	���87=�A_*


acc��>�U�       ��2	���87=�A`*

val_loss{�?n��       `/�#	���87=�A`*

val_acc�}�>͞
)       ��-	���87=�A`*

loss!�?D�W�       ��(	Y��87=�A`*


accρ�>w�       ��2	�t�87=�Aa*

val_lossN�?ʶ�       `/�#	v�87=�Aa*

val_accб�>��B2       ��-	�v�87=�Aa*

loss�x?��       ��(	�v�87=�Aa*


acc��>���g       ��2	yǸ87=�Ab*

val_loss>^?��       `/�#	ɸ87=�Ab*

val_acc��>��       ��-	�ɸ87=�Ab*

loss�c?}�2�       ��(	yʸ87=�Ab*


acc�Q�>d�KZ       ��2	���87=�Ac*

val_loss ?�A.�       `/�#	���87=�Ac*

val_acc� �>��K       ��-	��87=�Ac*

lossi7?�qb�       ��(	z��87=�Ac*


acc1��>-�	V