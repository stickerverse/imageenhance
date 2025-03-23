import torch
import torch.nn as nn
import torchvision
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Adapts pre-trained TorchVision models to work with the Neural Enhancement Pipeline.
    This class handles the conversion of pre-trained TorchVision models to the
    custom architectures used in the enhancement pipeline.
    """
    
    @staticmethod
    def adapt_super_resolution_model(state_dict, target_model):
        """
        Adapt a pre-trained ResNet to the super-resolution network.
        
        Args:
            state_dict: The state dict from a pre-trained ResNet
            target_model: The super-resolution model to adapt to
            
        Returns:
            The adapted state_dict
        """
        try:
            # We'll initialize the target model with random weights
            target_state_dict = target_model.state_dict()
            
            # Copy the first convolutional layer weights
            if 'conv1.weight' in state_dict and 'conv_first.weight' in target_state_dict:
                # Adjust channels if necessary
                src_weight = state_dict['conv1.weight']
                target_shape = target_state_dict['conv_first.weight'].shape
                
                if src_weight.shape[1] != target_shape[1]:  # Input channels mismatch
                    # Repeat channels to match target shape
                    src_weight = src_weight.repeat(1, target_shape[1] // src_weight.shape[1], 1, 1)
                
                # Resize kernel if necessary
                if src_weight.shape[2:] != target_shape[2:]:
                    src_weight = nn.functional.interpolate(
                        src_weight, size=target_shape[2:], mode='bilinear', align_corners=False
                    )
                
                target_state_dict['conv_first.weight'] = src_weight
            
            logger.info("Adapted super-resolution model successfully")
            return target_state_dict
        
        except Exception as e:
            logger.error(f"Error adapting super-resolution model: {str(e)}")
            return target_model.state_dict()  # Return original state dict on failure
    
    @staticmethod
    def adapt_denoising_model(state_dict, target_model):
        """
        Adapt a pre-trained ResNet to the denoising network.
        
        Args:
            state_dict: The state dict from a pre-trained ResNet
            target_model: The denoising model to adapt to
            
        Returns:
            The adapted state_dict
        """
        try:
            # We'll initialize the target model with random weights
            target_state_dict = target_model.state_dict()
            
            # For denoising model, we can try to copy encoder weights from ResNet
            if 'conv1.weight' in state_dict and 'enc1.0.weight' in target_state_dict:
                src_weight = state_dict['conv1.weight']
                target_shape = target_state_dict['enc1.0.weight'].shape
                
                if src_weight.shape[1] != target_shape[1]:  # Input channels mismatch
                    src_weight = src_weight.repeat(1, target_shape[1] // src_weight.shape[1], 1, 1)
                
                # Resize kernel if necessary
                if src_weight.shape[2:] != target_shape[2:]:
                    src_weight = nn.functional.interpolate(
                        src_weight, size=target_shape[2:], mode='bilinear', align_corners=False
                    )
                
                target_state_dict['enc1.0.weight'] = src_weight
            
            logger.info("Adapted denoising model successfully")
            return target_state_dict
        
        except Exception as e:
            logger.error(f"Error adapting denoising model: {str(e)}")
            return target_model.state_dict()  # Return original state dict on failure
    
    @staticmethod
    def adapt_segmentation_model(state_dict, target_model):
        """
        Adapt a pre-trained segmentation model to our UNet architecture.
        
        Args:
            state_dict: The state dict from a pre-trained FCN model
            target_model: The segmentation model to adapt to
            
        Returns:
            The adapted state_dict
        """
        try:
            # Initialize target model with random weights
            target_state_dict = target_model.state_dict()
            
            # Here we can try to adapt some weights from the FCN model
            # This is a simplified example - real adaptation would be more complex
            if 'backbone.conv1.weight' in state_dict and 'enc1.0.weight' in target_state_dict:
                src_weight = state_dict['backbone.conv1.weight']
                target_shape = target_state_dict['enc1.0.weight'].shape
                
                if src_weight.shape[1] != target_shape[1]:  # Input channels mismatch
                    src_weight = src_weight.repeat(1, target_shape[1] // src_weight.shape[1], 1, 1)
                
                # Resize kernel if necessary
                if src_weight.shape[2:] != target_shape[2:]:
                    src_weight = nn.functional.interpolate(
                        src_weight, size=target_shape[2:], mode='bilinear', align_corners=False
                    )
                
                target_state_dict['enc1.0.weight'] = src_weight
            
            logger.info("Adapted segmentation model successfully")
            return target_state_dict
        
        except Exception as e:
            logger.error(f"Error adapting segmentation model: {str(e)}")
            return target_model.state_dict()  # Return original state dict on failure
    
    @staticmethod
    def adapt_depth_model(state_dict, target_model):
        """
        Adapt a pre-trained ResNet to the depth estimation network.
        
        Args:
            state_dict: The state dict from a pre-trained ResNet
            target_model: The depth estimation model to adapt to
            
        Returns:
            The adapted state_dict
        """
        try:
            # Initialize target model with random weights
            target_state_dict = target_model.state_dict()
            
            # Copy the first convolutional layer weights for encoder
            if 'conv1.weight' in state_dict and 'enc1.0.weight' in target_state_dict:
                src_weight = state_dict['conv1.weight']
                target_shape = target_state_dict['enc1.0.weight'].shape
                
                if src_weight.shape[1] != target_shape[1]:  # Input channels mismatch
                    src_weight = src_weight.repeat(1, target_shape[1] // src_weight.shape[1], 1, 1)
                
                # Resize kernel if necessary
                if src_weight.shape[2:] != target_shape[2:]:
                    src_weight = nn.functional.interpolate(
                        src_weight, size=target_shape[2:], mode='bilinear', align_corners=False
                    )
                
                target_state_dict['enc1.0.weight'] = src_weight
            
            logger.info("Adapted depth estimation model successfully")
            return target_state_dict
        
        except Exception as e:
            logger.error(f"Error adapting depth estimation model: {str(e)}")
            return target_model.state_dict()  # Return original state dict on failure
    
    @staticmethod
    def adapt_style_transfer_model(state_dict, target_model):
        """
        Adapt a pre-trained ResNet to the style transfer network.
        
        Args:
            state_dict: The state dict from a pre-trained ResNet
            target_model: The style transfer model to adapt to
            
        Returns:
            The adapted state_dict
        """
        try:
            # Initialize target model with random weights
            target_state_dict = target_model.state_dict()
            
            # Copy the first convolutional layer weights for initial layer
            if 'conv1.weight' in state_dict and 'initial.0.weight' in target_state_dict:
                src_weight = state_dict['conv1.weight']
                target_shape = target_state_dict['initial.0.weight'].shape
                
                if src_weight.shape[1] != target_shape[1]:  # Input channels mismatch
                    src_weight = src_weight.repeat(1, target_shape[1] // src_weight.shape[1], 1, 1)
                
                # Resize kernel if necessary
                if src_weight.shape[2:] != target_shape[2:]:
                    src_weight = nn.functional.interpolate(
                        src_weight, size=target_shape[2:], mode='bilinear', align_corners=False
                    )
                
                target_state_dict['initial.0.weight'] = src_weight
            
            logger.info("Adapted style transfer model successfully")
            return target_state_dict
        
        except Exception as e:
            logger.error(f"Error adapting style transfer model: {str(e)}")
            return target_model.state_dict()  # Return original state dict on failure
    
    @staticmethod
    def adapt_object_detection_model(state_dict, target_model):
        """
        Adapt a pre-trained object detection model to our custom architecture.
        
        Args:
            state_dict: The state dict from a pre-trained Faster R-CNN
            target_model: The object detection model to adapt to
            
        Returns:
            The adapted state_dict
        """
        try:
            # Initialize target model with random weights
            target_state_dict = target_model.state_dict()
            
            # Adapt the backbone's first layer
            if 'backbone.body.conv1.weight' in state_dict and 'backbone.0.weight' in target_state_dict:
                src_weight = state_dict['backbone.body.conv1.weight']
                target_shape = target_state_dict['backbone.0.weight'].shape
                
                if src_weight.shape[1] != target_shape[1]:  # Input channels mismatch
                    src_weight = src_weight.repeat(1, target_shape[1] // src_weight.shape[1], 1, 1)
                
                # Resize kernel if necessary
                if src_weight.shape[2:] != target_shape[2:]:
                    src_weight = nn.functional.interpolate(
                        src_weight, size=target_shape[2:], mode='bilinear', align_corners=False
                    )
                
                target_state_dict['backbone.0.weight'] = src_weight
            
            logger.info("Adapted object detection model successfully")
            return target_state_dict
        
        except Exception as e:
            logger.error(f"Error adapting object detection model: {str(e)}")
            return target_model.state_dict()  # Return original state dict on failure