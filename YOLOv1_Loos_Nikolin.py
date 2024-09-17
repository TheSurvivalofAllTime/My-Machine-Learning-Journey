
import tensorflow as tf

class Yolo_Loss_nikolin1(object):

    def __init__(self, S=7, B=2, num_classes=3, lambda_coord=5, lambda_noobj=0.5):
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, box1, box2):
        # Convert from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
        box1_x1 = box1[..., 0] - box1[..., 2] / 2.0
        box1_y1 = box1[..., 1] - box1[..., 3] / 2.0
        box1_x2 = box1[..., 0] + box1[..., 2] / 2.0
        box1_y2 = box1[..., 1] + box1[..., 3] / 2.0

        box2_x1 = box2[..., 0] - box2[..., 2] / 2.0
        box2_y1 = box2[..., 1] - box2[..., 3] / 2.0
        box2_x2 = box2[..., 0] + box2[..., 2] / 2.0
        box2_y2 = box2[..., 1] + box2[..., 3] / 2.0

        # Determine the coordinates of the intersection rectangle
        inter_x1 = tf.maximum(box1_x1, box2_x1)
        inter_y1 = tf.maximum(box1_y1, box2_y1)
        inter_x2 = tf.minimum(box1_x2, box2_x2)
        inter_y2 = tf.minimum(box1_y2, box2_y2)

        # Compute the area of the intersection rectangle
        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)

        # Compute the area of both the prediction and ground-truth rectangles
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        # Compute the IoU by dividing the intersection area by the union area
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / (union_area + 1e-10)

        return iou


    
    # def compute_iou(self, box1, box2):
        
    #     x1 = tf.maximum(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    #     # the leftmost part of the intersection will be at the larger x-coordinate (i.e., further to the right).
    #     #  This is because the intersection can only start where both boxes begin to overlap.
    #     y1 = tf.maximum(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    #     # Similarly for y1

    #     x2 = tf.minimum(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    #     #x-coordinate of the right edge of the intersection area is the leftmost (farthest to the left) right edge of the two boxes.
    #     #  This guarantees the intersection ends where both boxes still overlap
    #     y2 = tf.minimum(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)
    #     #Similarly for 

    #     intersection_area = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    #     box1_area = box1[..., 2] * box1[..., 3]
    #     box2_area = box2[..., 2] * box2[..., 3]
    #     union_area = box1_area + box2_area - intersection_area
    #     iou = intersection_area / (union_area + 1e-8)
    #     return iou

    def keep_or_eliminate_box(self, y_true, y_pred):
        keep_or_eliminate_box1 = 0.0
        keep_or_eliminate_box2 = 0.0
        obj_mask1 = tf.expand_dims(y_true[..., 7], axis=-1)
        obj_mask2 = tf.expand_dims(y_true[..., 12], axis=-1)

        box1_pred = y_pred[..., 3:7] * obj_mask1
        box2_pred = y_pred[..., 8:12] * obj_mask2
        box1_true = y_true[..., 3:7] * obj_mask1
        box2_true = y_true[..., 8:12] * obj_mask2

        box1_iou = self.compute_iou(box1_true, box1_pred)
        box1_iou = float(tf.reduce_sum(box1_iou))
        #box1_iou = tf.reduce_sum(box1_iou)

        box2_iou = self.compute_iou(box2_true, box2_pred)
        box2_iou = float(tf.reduce_sum(box2_iou))
        #box2_iou = tf.reduce_sum(box2_iou)

        # keep_or_eliminate_box1 = tf.where(box1_iou >= box2_iou, 1.0, 0.0)
        # keep_or_eliminate_box2 = tf.where(box1_iou < box2_iou, 1.0, 0.0)

        
        if box1_iou >= box2_iou:
            keep_or_eliminate_box1 = 1.0
        else:
            keep_or_eliminate_box2 = 1.0

        # print('keep_or_eliminate_box1: ', keep_or_eliminate_box1)
        # print('keep_or_eliminate_box2: ', keep_or_eliminate_box2)
        # print(box1_iou, box1_iou, '<----')

        return obj_mask1,  keep_or_eliminate_box1, keep_or_eliminate_box2


    def localization_metric(self, y_true, y_pred):

        #y_true = tf.cast(y_true, tf.float32)
        #y_pred = tf.cast(y_pred, tf.float32)
        
        box1_pred = y_pred[..., 3:7]
        box2_pred = y_pred[..., 8:12]
        box1_true = y_true[..., 3:7]
        box2_true = y_true[..., 8:12]

        # Set the box with low iou score to zero
        obj_mask, keep_or_eliminate_box1, keep_or_eliminate_box2 = self.keep_or_eliminate_box(y_true, y_pred)

        predicted_box = obj_mask*(keep_or_eliminate_box1 * box1_pred + keep_or_eliminate_box2 * box2_pred)

        xy_pred_coordinates = predicted_box[..., 0:2]
        wh_pred_sqrt = tf.sign(predicted_box[...,2:4])* tf.sqrt(tf.abs(predicted_box[...,2:4]+1e-6))


        #predicted_box[...,2:4] = tf.sign(predicted_box[...,2:4])* tf.sqrt(tf.abs(predicted_box[...,2:4]+1e-6))

        box_true = obj_mask * box1_true
        wh_true_sqrt= tf.sqrt(box_true[..., 2:4])
        xy_true_coordinates = box_true[..., 0:2]

        xy_sum_loss = tf.reduce_sum( tf.square(xy_true_coordinates - xy_pred_coordinates) )
        wh_sqrt_sum_losses = tf.reduce_sum(   tf.square(wh_true_sqrt-wh_pred_sqrt))
        loss_box = xy_sum_loss+ wh_sqrt_sum_losses

        return loss_box * self.lambda_coord

    def confidence_loss_metric(self, y_true, y_pred):

        obj_mask, keep_or_eliminate_box1, keep_or_eliminate_box2 = self.keep_or_eliminate_box(y_true, y_pred)
        no_obj_mask = 1-obj_mask


        pred_confidence_box = keep_or_eliminate_box1*y_pred[..., 7:8] + keep_or_eliminate_box2* y_pred[..., 12:13]

        pred_confidence_box = obj_mask * pred_confidence_box
        true_confidence = obj_mask * y_true[..., 7:8]

        object_confidence_loss = tf.reduce_sum( tf.square(true_confidence - pred_confidence_box) )

        no_object_confidence_loss_B1 = no_obj_mask * tf.reduce_sum( tf.square(y_true[..., 7:8]- y_pred[..., 7:8]) )
        no_object_confidence_loss_B2 = no_obj_mask * tf.reduce_sum( tf.square(y_true[..., 12:13]- y_pred[..., 12:13]) )
        both_B1_b2 = no_object_confidence_loss_B1 + no_object_confidence_loss_B2

        return self.lambda_noobj * both_B1_b2 + object_confidence_loss


    def classification_loss_metric(self, y_true, y_pred):
        class_probs_true = y_true[..., :self.num_classes]
        class_probs_pred = y_pred[..., :self.num_classes]


        obj_mask = tf.expand_dims(y_true[..., 7], axis=-1) 

        #obj_mask = y_true[..., 7]  # Only consider the cells with objects
        classification_loss = tf.reduce_sum(obj_mask * tf.square(class_probs_true - class_probs_pred))
        return classification_loss
    

    def TotalLoss(self, y_true, y_pred):
        total_loss = (self.localization_metric(y_true, y_pred) +
                      self.confidence_loss_metric(y_true, y_pred) +
                      self.classification_loss_metric(y_true, y_pred))
        return total_loss




# def yolo_loss_nikolin1_hey(y_true, y_pred, S=7, B=2, C=3, lambda_coord=5, lambda_noobj=0.5):
#     """
#     Compute the YOLOv1 loss.
    
#     Arguments:
#     y_true -- Ground truth tensor of shape (batch_size, S, S, B * 5 + C)
#     y_pred -- Predicted tensor of shape (batch_size, S, S, B * 5 + C)
#     S -- Number of grid cells (default 7)
#     B -- Number of bounding boxes per grid cell (default 2)
#     C -- Number of classes (default 3)
#     lambda_coord -- Weight for localization loss
#     lambda_noobj -- Weight for confidence loss for no object
    
#     Returns:
#     loss -- The total YOLOv1 loss
#     """

#     # Reshape y_true and y_pred to be able to extract individual components
#     # Ground thruth
#     y_true = tf.reshape(y_true, (-1, S, S, B * 5 + C))
#     # Predicted 
#     y_pred = tf.reshape(y_pred, (-1, S, S, B * 5 + C))


#     # 1_{i,j}^{obj}
#     box_one_and_two_obj_mask = y_true[..., 7]  # Mask indicating where the first and second bounding box is responsible
    
#     # Extract ground truth components
#     class_probs_true = y_true[..., :C]
#     # Extract class probabilities
#     class_probs_pred = y_pred[..., :C]
    
#     # Extract x, y, w, h, and confidence for the first bounding box
#     box1_x_true =  y_true[..., 3] # ground truth x_center_grid for box1 
#     box1_y_true = y_true[..., 4] # ground truth y_center_grid for box1
#     box1_w_true = y_true[..., 5] # ground truth width of the object for box1
#     box1_h_true = y_true[..., 6] # ground truth height of the object for box1
#     box1_conf_true = y_true[..., 7] # ground truth confidence score for box1
    

#     box1_x_pred = y_pred[..., 3] # # Predicted x_center for box1
#     box1_y_pred = y_pred[..., 4] # # Predicted y_center for box1 
#     box1_w_pred = y_pred[..., 5] # Predicted width  for box1
#     box1_h_pred = y_pred[..., 6] # Predicted height  for box1
#     box1_conf_pred = y_pred[..., 7] #  Predicted confidence score  for box 1
    
#     # Extract x, y, w, h, and confidence for the second bounding box
#     box2_x_true = y_true[..., 8] # ground truth x_center_grid for box2
#     box2_y_true = y_true[..., 9] # ground truth y_center_grid for box2
#     box2_w_true = y_true[..., 10] # ground truth width of the object for box2
#     box2_h_true = y_true[..., 11] # ground truth width of the object for box2
#     box2_conf_true = y_true[..., 12] # ground truth confidence score for box2
    
#     box2_x_pred = y_pred[..., 8] #  Predicted x_center for box2
#     box2_y_pred = y_pred[..., 9] #  Predicted y_center for box2
#     box2_w_pred = y_pred[..., 10] # Predicted width  for box2
#     box2_h_pred = y_pred[..., 11] # Predicted height  for box2
#     box2_conf_pred = y_pred[..., 12] # Predicted confidence score for box 2


#     # Prepare width and height for localization loss by applying square root
#     # We add 1/1000000 to prevent taking the square root of zero, which could lead to numerical instability 
#     # or errors in the optimization process or result in NaN (Not a Number) values during training.

 
#     box1_w_true_sqrt = tf.sqrt(box1_w_true + 1e-6)
#     box1_h_true_sqrt = tf.sqrt(box1_h_true + 1e-6)
#     box1_w_pred_sqrt = tf.sqrt(box1_w_pred + 1e-6)
#     box1_h_pred_sqrt = tf.sqrt(box1_h_pred + 1e-6)

#     box2_w_true_sqrt = tf.sqrt(box2_w_true + 1e-6)
#     box2_h_true_sqrt = tf.sqrt(box2_h_true + 1e-6)
#     box2_w_pred_sqrt = tf.sqrt(box2_w_pred + 1e-6)
#     box2_h_pred_sqrt = tf.sqrt(box2_h_pred + 1e-6)


#     # Calculate the localization loss (for x, y)
#     # j=1, also the first box and 1_{i,j}^{obj} = box_one_and_two_obj_mask
#     Sum_loss_box1_xy = tf.reduce_sum( box_one_and_two_obj_mask*(tf.square(box1_x_true - box1_x_pred )  + tf.square(box1_y_true-box1_y_pred) ))
#     # j=2, also the second box and 1_{i,j}^{obj} = box_one_and_two_obj_mask
#     Sum_loss_box2_xy = tf.reduce_sum( box_one_and_two_obj_mask * (tf.square(box2_x_true - box2_x_pred) + tf.square(box2_y_true - box2_y_pred)) )

#     localization_loss_xy = lambda_coord*(Sum_loss_box1_xy+ Sum_loss_box2_xy)

#     # Calculate the localization loss ( width, height)
#     # j=1, also the first box and 1_{i,j}^{obj} = box_one_and_two_obj_mask
#     Sum_loss_box1_wh = tf.reduce_sum( box_one_and_two_obj_mask* (tf.square(box1_w_true_sqrt-box1_w_pred_sqrt)+ tf.square(box1_h_true_sqrt-box1_h_pred_sqrt))  )
#     # j=2, also the first box and 1_{i,j}^{obj} = box_one_and_two_obj_mask
#     Sum_loss_box2_wh = tf.reduce_sum(box_one_and_two_obj_mask *(tf.square(box2_w_true_sqrt-box2_w_pred_sqrt)+ tf.square(box2_h_true_sqrt-box2_h_pred_sqrt)))

#     localization_loss_wh = lambda_coord*(Sum_loss_box1_wh+Sum_loss_box2_wh)

#     localization_loss_total = localization_loss_xy + localization_loss_wh

    


#     confidence_loss_box1_obj = tf.reduce_sum( tf.square(box_one_and_two_obj_mask * (box1_conf_true - box1_conf_pred) ))
#     # In the tensor at position 7, there is the confidence score: 1 if an object is present, or 0 if no object is present.
#     # We aren't sure that the predicted confidence score in a grid cell without an object would be zero as required by the formula.
#     # The true confidence score is 1 for the object and 0 for non-objects. The model could predict a non-zero confidence score for non-objects,
#     # and taking the squared difference between the true confidence scores and the wrongly predicted confidence score could incorrectly increase the loss.
#     # Multiplying by the ground truth confidence score ensures that only relevant grid cells (those with objects) contribute to the loss.
#     confidence_loss_box2_obj = tf.reduce_sum( tf.square(box_one_and_two_obj_mask * (box2_conf_true - box2_conf_pred) ))
#     confidence_loss_obj = confidence_loss_box1_obj+ confidence_loss_box2_obj


#     confidence_loss_noobj = lambda_noobj * (
#         tf.reduce_sum(tf.square((1 - box1_conf_true) * box1_conf_pred)) +
#         tf.reduce_sum(tf.square((1 - box2_conf_true) * box2_conf_pred))
#     )

#     total_loss = localization_loss_total + confidence_loss_obj + confidence_loss_noobj
#     return total_loss, localization_loss_total, confidence_loss_obj, confidence_loss_noobj
