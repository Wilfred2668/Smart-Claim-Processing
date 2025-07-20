"""
Computer Vision Analysis Module
Handles image processing and property condition assessment
"""

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64


class VisionAnalyzer:
    """Analyzes property images for condition assessment and hazard detection"""
    
    def __init__(self):
        # Load YOLO model for object detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
        except:
            self.yolo_model = None
        
        # Define property-specific detection categories
        self.property_objects = [
            'person', 'car', 'truck', 'bicycle', 'motorcycle',  # People and vehicles
            'chair', 'couch', 'bed', 'dining table', 'tv', 'laptop',  # Furniture
            'refrigerator', 'oven', 'microwave', 'sink',  # Appliances
            'window', 'door', 'stairs', 'fire hydrant', 'stop sign'  # Building features
        ]
        
        # Hazard indicators
        self.hazard_indicators = {
            'water_damage': ['dark_stains', 'discoloration', 'mold_growth'],
            'structural': ['cracks', 'sagging', 'uneven_surfaces'],
            'electrical': ['exposed_wires', 'damaged_outlets', 'overloaded_circuits'],
            'fire_hazard': ['flammable_materials', 'blocked_exits', 'smoke_damage'],
            'maintenance': ['peeling_paint', 'broken_windows', 'damaged_siding']
        }
    
    def analyze_image(self, image_input) -> Dict:
        """Main method to analyze property images"""
        try:
            # Handle both file paths and UploadedFile objects
            if hasattr(image_input, 'name'):  # UploadedFile object
                # Load image from uploaded file
                image = Image.open(image_input)
                image_np = np.array(image)
                
                # Convert to RGB if needed
                if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                elif len(image_np.shape) == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:  # File path string
                # Load image from file path
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not read image: {image_input}")
                image_np = image
            
            # Convert to RGB for analysis
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Perform various analyses
            analysis_results = {
                'image_quality': self._assess_image_quality(image_np),
                'property_condition': self._assess_property_condition(image_rgb),
                'hazards_detected': self._detect_hazards(image_rgb),
                'object_detection': self._detect_objects(image_rgb),
                'color_analysis': self._analyze_colors(image_rgb),
                'structural_analysis': self._analyze_structure(image_rgb),
                'overall_score': 0.0
            }
            
            # Calculate overall condition score
            analysis_results['overall_score'] = self._calculate_overall_score(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            # Return a safe default structure if anything goes wrong
            print(f"Error analyzing image: {str(e)}")
            return {
                'image_quality': {
                    'brightness': 128,
                    'contrast': 50,
                    'sharpness': 100,
                    'blur_score': 0.5,
                    'quality_score': 0.5,
                    'issues': ['Image processing failed']
                },
                'property_condition': {
                    'condition_score': 0.5,
                    'issues': ['Analysis failed'],
                    'dark_ratio': 0.5,
                    'color_uniformity': 50,
                    'edge_density': 0.05
                },
                'hazards_detected': [],
                'object_detection': [],
                'color_analysis': {
                    'mean_rgb': [128, 128, 128],
                    'std_rgb': [50, 50, 50],
                    'mean_hsv': [0, 0, 128],
                    'std_hsv': [0, 0, 50],
                    'dominant_colors': []
                },
                'structural_analysis': {
                    'horizontal_lines': 0,
                    'vertical_lines': 0,
                    'total_lines': 0,
                    'edge_density': 0.05
                },
                'overall_score': 0.5,
                'error': str(e)
            }
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Assess the quality of the input image"""
        # Calculate basic image statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.std(gray)
        
        # Sharpness (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Blur detection
        blur_score = self._detect_blur(gray)
        
        # Quality assessment
        quality_score = 0.0
        quality_issues = []
        
        if brightness < 50:
            quality_issues.append("Low brightness")
            quality_score -= 0.2
        elif brightness > 200:
            quality_issues.append("Overexposed")
            quality_score -= 0.2
        
        if contrast < 30:
            quality_issues.append("Low contrast")
            quality_score -= 0.2
        
        if laplacian_var < 100:
            quality_issues.append("Blurry image")
            quality_score -= 0.3
        
        if blur_score > 0.8:
            quality_issues.append("Significant blur detected")
            quality_score -= 0.3
        
        quality_score = max(0.0, min(1.0, quality_score + 1.0))
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': laplacian_var,
            'blur_score': blur_score,
            'quality_score': quality_score,
            'issues': quality_issues
        }
    
    def _detect_blur(self, gray_image: np.ndarray) -> float:
        """Detect blur in the image using FFT"""
        # Apply FFT
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate high-frequency content
        rows, cols = gray_image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create high-pass filter
        high_freq = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
        total_freq = magnitude_spectrum
        
        # Calculate blur score
        high_freq_energy = np.sum(high_freq)
        total_energy = np.sum(total_freq)
        
        blur_score = 1 - (high_freq_energy / total_energy) if total_energy > 0 else 1.0
        
        return blur_score
    
    def _assess_property_condition(self, image: np.ndarray) -> Dict:
        """Assess overall property condition from visual analysis"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Analyze color distribution
        color_analysis = self._analyze_colors(image)
        
        # Detect potential issues
        issues = []
        condition_score = 1.0
        
        # Check for dark areas (potential water damage)
        dark_pixels = np.sum(hsv[:, :, 2] < 50)
        total_pixels = image.shape[0] * image.shape[1]
        dark_ratio = dark_pixels / total_pixels
        
        if dark_ratio > 0.3:
            issues.append("Large dark areas detected")
            condition_score -= 0.2
        
        # Check for color uniformity (indicates maintenance)
        color_std = np.std(image, axis=(0, 1))
        if np.mean(color_std) > 60:
            issues.append("Inconsistent coloring")
            condition_score -= 0.1
        
        # Check for structural patterns
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        if edge_density > 0.1:
            issues.append("High edge density - potential structural complexity")
        
        return {
            'condition_score': max(0.0, condition_score),
            'issues': issues,
            'dark_ratio': dark_ratio,
            'color_uniformity': np.mean(color_std),
            'edge_density': edge_density
        }
    
    def _detect_hazards(self, image: np.ndarray) -> List[Dict]:
        """Detect potential hazards in the image"""
        hazards = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Water damage detection (dark, discolored areas)
        water_damage = self._detect_water_damage(hsv, gray)
        if water_damage:
            hazards.append({
                'type': 'water_damage',
                'confidence': water_damage['confidence'],
                'severity': water_damage['severity'],
                'description': 'Potential water damage detected'
            })
        
        # Structural issues detection
        structural_issues = self._detect_structural_issues(gray)
        if structural_issues:
            hazards.append({
                'type': 'structural',
                'confidence': structural_issues['confidence'],
                'severity': structural_issues['severity'],
                'description': 'Potential structural issues detected'
            })
        
        # Electrical hazards
        electrical_hazards = self._detect_electrical_hazards(image)
        if electrical_hazards:
            hazards.append({
                'type': 'electrical',
                'confidence': electrical_hazards['confidence'],
                'severity': electrical_hazards['severity'],
                'description': 'Potential electrical hazards detected'
            })
        
        return hazards
    
    def _detect_water_damage(self, hsv: np.ndarray, gray: np.ndarray) -> Optional[Dict]:
        """Detect water damage patterns"""
        # Look for dark, discolored areas
        dark_mask = hsv[:, :, 2] < 80
        saturation_mask = hsv[:, :, 1] < 50
        
        # Combine masks
        water_damage_mask = dark_mask & saturation_mask
        
        # Calculate affected area
        affected_ratio = np.sum(water_damage_mask) / water_damage_mask.size
        
        if affected_ratio > 0.05:  # More than 5% of image
            severity = min(1.0, affected_ratio * 10)  # Scale to 0-1
            confidence = min(0.8, affected_ratio * 8)  # Scale to 0-0.8
            
            return {
                'confidence': confidence,
                'severity': severity,
                'affected_area': affected_ratio
            }
        
        return None
    
    def _detect_structural_issues(self, gray: np.ndarray) -> Optional[Dict]:
        """Detect structural issues like cracks"""
        # Edge detection for cracks
        edges = cv2.Canny(gray, 30, 100)
        
        # Morphological operations to connect crack lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for crack-like patterns
        crack_score = 0
        for contour in contours:
            if len(contour) > 10:  # Minimum contour size
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Check if contour is long and thin (crack-like)
                    aspect_ratio = area / (perimeter ** 2)
                    if aspect_ratio < 0.01:  # Long and thin
                        crack_score += area
        
        # Normalize score
        total_area = gray.shape[0] * gray.shape[1]
        crack_ratio = crack_score / total_area
        
        if crack_ratio > 0.001:  # Significant cracks detected
            severity = min(1.0, crack_ratio * 1000)
            confidence = min(0.7, crack_ratio * 700)
            
            return {
                'confidence': confidence,
                'severity': severity,
                'crack_ratio': crack_ratio
            }
        
        return None
    
    def _detect_electrical_hazards(self, image: np.ndarray) -> Optional[Dict]:
        """Detect electrical hazards"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Look for exposed wires (dark, thin lines)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for electrical outlets (rectangular shapes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        electrical_score = 0
        for contour in contours:
            if len(contour) > 5:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's rectangular (electrical outlet)
                if len(approx) == 4:
                    electrical_score += cv2.contourArea(contour)
        
        # Normalize score
        total_area = image.shape[0] * image.shape[1]
        electrical_ratio = electrical_score / total_area
        
        if electrical_ratio > 0.0001:
            severity = min(1.0, electrical_ratio * 10000)
            confidence = min(0.6, electrical_ratio * 6000)
            
            return {
                'confidence': confidence,
                'severity': severity,
                'electrical_ratio': electrical_ratio
            }
        
        return None
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect objects in the image using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        # Only include property-relevant objects
                        if class_name in self.property_objects and confidence > 0.3:
                            detections.append({
                                'object': class_name,
                                'confidence': confidence,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'area': (x2 - x1) * (y2 - y1)
                            })
            
            return detections
        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def _analyze_colors(self, image: np.ndarray) -> Dict:
        """Analyze color distribution in the image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        color_stats = {
            'mean_rgb': np.mean(image, axis=(0, 1)).tolist(),
            'std_rgb': np.std(image, axis=(0, 1)).tolist(),
            'mean_hsv': np.mean(hsv, axis=(0, 1)).tolist(),
            'std_hsv': np.std(hsv, axis=(0, 1)).tolist(),
        }
        
        # Analyze dominant colors
        pixels = image.reshape(-1, 3)
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
            color_stats['dominant_colors'] = dominant_colors
        except:
            color_stats['dominant_colors'] = []
        
        return color_stats
    
    def _analyze_structure(self, image: np.ndarray) -> Dict:
        """Analyze structural elements in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        # Analyze lines for structural patterns
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 10:  # Horizontal
                    horizontal_lines += 1
                elif abs(angle - 90) < 10:  # Vertical
                    vertical_lines += 1
        
        return {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'total_lines': horizontal_lines + vertical_lines,
            'edge_density': np.sum(edges > 0) / edges.size
        }
    
    def _calculate_overall_score(self, analysis_results: Dict) -> float:
        """Calculate overall condition score from all analyses"""
        score = 1.0
        
        # Image quality impact
        quality_score = analysis_results['image_quality']['quality_score']
        score *= quality_score
        
        # Property condition impact
        condition_score = analysis_results['property_condition']['condition_score']
        score *= condition_score
        
        # Hazard impact
        hazards = analysis_results['hazards_detected']
        for hazard in hazards:
            score *= (1 - hazard['severity'] * 0.3)  # Reduce score based on hazard severity
        
        return max(0.0, min(1.0, score))
    
    def create_analysis_visualization(self, image_input, analysis_results: Dict) -> str:
        """Create a visualization of the analysis results"""
        # Handle both file paths and UploadedFile objects
        if hasattr(image_input, 'name'):  # UploadedFile object
            # Load image from uploaded file
            image = Image.open(image_input)
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:  # File path string
            # Load image from file path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image: {image_input}")
            image_np = image
        
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Quality analysis
        quality = analysis_results['image_quality']
        axes[0, 1].text(0.1, 0.9, f"Quality Score: {quality['quality_score']:.2f}", 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].text(0.1, 0.8, f"Brightness: {quality['brightness']:.1f}", 
                       transform=axes[0, 1].transAxes, fontsize=10)
        axes[0, 1].text(0.1, 0.7, f"Contrast: {quality['contrast']:.1f}", 
                       transform=axes[0, 1].transAxes, fontsize=10)
        axes[0, 1].text(0.1, 0.6, f"Sharpness: {quality['sharpness']:.1f}", 
                       transform=axes[0, 1].transAxes, fontsize=10)
        axes[0, 1].set_title('Image Quality Analysis')
        axes[0, 1].axis('off')
        
        # Hazards visualization
        axes[1, 0].imshow(image_rgb)
        for hazard in analysis_results['hazards_detected']:
            # Add hazard annotations
            axes[1, 0].text(0.1, 0.9 - len(analysis_results['hazards_detected'])*0.1, 
                           f"{hazard['type']}: {hazard['severity']:.2f}", 
                           transform=axes[1, 0].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
        axes[1, 0].set_title('Hazards Detected')
        axes[1, 0].axis('off')
        
        # Overall score
        overall_score = analysis_results['overall_score']
        axes[1, 1].text(0.5, 0.5, f"Overall Score: {overall_score:.2f}", 
                       transform=axes[1, 1].transAxes, fontsize=16, ha='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Overall Assessment')
        axes[1, 1].axis('off')
        
        # Save to base64 string
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64 