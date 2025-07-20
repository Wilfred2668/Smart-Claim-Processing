"""
Enhanced Risk Assessment Engine
Advanced statistical analysis and correlation-based risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import math
from scipy import stats
from collections import defaultdict


class EnhancedRiskAssessor:
    """Advanced risk assessor with statistical analysis and correlation detection"""
    
    def __init__(self):
        # Load enhanced underwriting guidelines with statistical parameters
        self.guidelines = self._load_enhanced_guidelines()
        
        # Statistical parameters for risk assessment
        self.statistical_params = {
            'confidence_level': 0.95,
            'outlier_threshold': 2.0,  # Standard deviations for outlier detection
            'correlation_threshold': 0.3,  # Minimum correlation to consider
            'sample_size_minimum': 10
        }
        
        # Enhanced risk weights with statistical confidence
        self.risk_weights = {
            'property_condition': {'weight': 0.25, 'confidence': 0.85},
            'hazards': {'weight': 0.30, 'confidence': 0.90},
            'property_age': {'weight': 0.15, 'confidence': 0.95},
            'location_risk': {'weight': 0.10, 'confidence': 0.80},
            'maintenance_issues': {'weight': 0.20, 'confidence': 0.75}
        }
        
        # Risk thresholds with confidence intervals
        self.risk_thresholds = {
            'low': {'threshold': 0.3, 'confidence_interval': 0.05},
            'medium': {'threshold': 0.6, 'confidence_interval': 0.08},
            'high': {'threshold': 0.8, 'confidence_interval': 0.10}
        }
        
        # Historical data patterns for comparison
        self.historical_patterns = self._load_historical_patterns()
    
    def _load_enhanced_guidelines(self) -> Dict:
        """Load enhanced underwriting guidelines with statistical parameters"""
        return {
            'property_condition': {
                'excellent': {
                    'score': 1.0, 
                    'risk_multiplier': 0.5,
                    'std_dev': 0.1,
                    'confidence_interval': 0.05
                },
                'good': {
                    'score': 0.8, 
                    'risk_multiplier': 0.7,
                    'std_dev': 0.15,
                    'confidence_interval': 0.08
                },
                'fair': {
                    'score': 0.6, 
                    'risk_multiplier': 1.0,
                    'std_dev': 0.2,
                    'confidence_interval': 0.12
                },
                'poor': {
                    'score': 0.3, 
                    'risk_multiplier': 1.5,
                    'std_dev': 0.25,
                    'confidence_interval': 0.15
                },
                'unknown': {
                    'score': 0.5, 
                    'risk_multiplier': 1.2,
                    'std_dev': 0.3,
                    'confidence_interval': 0.20
                }
            },
            'hazards': {
                'water_damage': {
                    'severity_high': {'risk': 0.8, 'frequency': 0.15, 'correlation': 0.6},
                    'severity_medium': {'risk': 0.5, 'frequency': 0.25, 'correlation': 0.4},
                    'severity_low': {'risk': 0.2, 'frequency': 0.35, 'correlation': 0.2}
                },
                'structural': {
                    'severity_high': {'risk': 0.9, 'frequency': 0.10, 'correlation': 0.8},
                    'severity_medium': {'risk': 0.6, 'frequency': 0.20, 'correlation': 0.6},
                    'severity_low': {'risk': 0.3, 'frequency': 0.30, 'correlation': 0.3}
                },
                'electrical': {
                    'severity_high': {'risk': 0.7, 'frequency': 0.12, 'correlation': 0.5},
                    'severity_medium': {'risk': 0.4, 'frequency': 0.22, 'correlation': 0.3},
                    'severity_low': {'risk': 0.2, 'frequency': 0.32, 'correlation': 0.1}
                },
                'fire_hazard': {
                    'severity_high': {'risk': 0.9, 'frequency': 0.08, 'correlation': 0.9},
                    'severity_medium': {'risk': 0.6, 'frequency': 0.18, 'correlation': 0.7},
                    'severity_low': {'risk': 0.3, 'frequency': 0.28, 'correlation': 0.4}
                },
                'maintenance': {
                    'severity_high': {'risk': 0.6, 'frequency': 0.20, 'correlation': 0.4},
                    'severity_medium': {'risk': 0.4, 'frequency': 0.30, 'correlation': 0.2},
                    'severity_low': {'risk': 0.2, 'frequency': 0.40, 'correlation': 0.1}
                }
            },
            'property_age': {
                'new': {'years': 0, 'risk_multiplier': 0.6, 'std_dev': 2.0},
                'recent': {'years': 10, 'risk_multiplier': 0.8, 'std_dev': 5.0},
                'established': {'years': 25, 'risk_multiplier': 1.0, 'std_dev': 10.0},
                'older': {'years': 50, 'risk_multiplier': 1.3, 'std_dev': 15.0},
                'historic': {'years': 100, 'risk_multiplier': 1.6, 'std_dev': 25.0}
            },
            'location_risk': {
                'flood_zone': {'risk': 0.8, 'frequency': 0.10, 'correlation': 0.7},
                'earthquake_zone': {'risk': 0.6, 'frequency': 0.15, 'correlation': 0.5},
                'hurricane_zone': {'risk': 0.7, 'frequency': 0.12, 'correlation': 0.6},
                'high_crime': {'risk': 0.5, 'frequency': 0.20, 'correlation': 0.4},
                'industrial_area': {'risk': 0.4, 'frequency': 0.25, 'correlation': 0.3}
            }
        }
    
    def _load_historical_patterns(self) -> Dict:
        """Load historical data patterns for statistical comparison"""
        return {
            'property_values': {
                'mean': 350000,
                'std_dev': 150000,
                'percentiles': [25, 50, 75, 90, 95],
                'percentile_values': [250000, 350000, 450000, 600000, 750000]
            },
            'square_footage': {
                'mean': 2000,
                'std_dev': 800,
                'percentiles': [25, 50, 75, 90, 95],
                'percentile_values': [1500, 2000, 2500, 3000, 3500]
            },
            'age_distribution': {
                'mean': 35,
                'std_dev': 20,
                'percentiles': [25, 50, 75, 90, 95],
                'percentile_values': [20, 35, 50, 65, 80]
            },
            'risk_score_distribution': {
                'mean': 0.45,
                'std_dev': 0.25,
                'percentiles': [25, 50, 75, 90, 95],
                'percentile_values': [0.25, 0.45, 0.65, 0.80, 0.90]
            }
        }
    
    def assess_risk_enhanced(self, data_input) -> Dict:
        """Enhanced risk assessment with statistical analysis"""
        try:
            # Combine data sources
            combined_data = self._combine_data_sources_enhanced(data_input)
            
            # Perform statistical analysis
            statistical_analysis = self._perform_statistical_analysis(combined_data)
            
            # Calculate correlation matrix
            correlation_analysis = self._calculate_correlations(combined_data)
            
            # Detect outliers
            outlier_analysis = self._detect_outliers(combined_data)
            
            # Calculate enhanced risk components
            risk_components = {
                'property_condition_risk': self._assess_property_condition_enhanced(combined_data),
                'hazard_risk': self._assess_hazard_risk_enhanced(combined_data),
                'age_risk': self._assess_age_risk_enhanced(combined_data),
                'location_risk': self._assess_location_risk_enhanced(combined_data),
                'maintenance_risk': self._assess_maintenance_risk_enhanced(combined_data)
            }
            
            # Calculate overall risk with confidence intervals
            overall_risk = self._calculate_overall_risk_enhanced(risk_components, statistical_analysis)
            
            # Determine risk category with statistical confidence
            risk_category = self._categorize_risk_enhanced(overall_risk)
            
            # Generate enhanced recommendations
            recommendations = self._generate_recommendations_enhanced(
                risk_components, overall_risk, statistical_analysis, outlier_analysis
            )
            
            # Create comprehensive assessment
            assessment = {
                'risk_score': overall_risk['score'],
                'risk_category': risk_category['category'],
                'confidence_interval': overall_risk['confidence_interval'],
                'statistical_significance': overall_risk['statistical_significance'],
                'risk_components': risk_components,
                'statistical_analysis': statistical_analysis,
                'correlation_analysis': correlation_analysis,
                'outlier_analysis': outlier_analysis,
                'recommendations': recommendations,
                'guideline_compliance': self._check_guideline_compliance_enhanced(combined_data),
                'assessment_timestamp': datetime.now().isoformat(),
                'confidence_score': self._calculate_confidence_enhanced(combined_data, statistical_analysis),
                'data_quality_score': self._assess_data_quality(combined_data)
            }
            
            return assessment
            
        except Exception as e:
            print(f"Error in enhanced risk assessment: {str(e)}")
            return self._get_default_assessment(str(e))
    
    def _combine_data_sources_enhanced(self, data_input) -> Dict:
        """Enhanced data combination with validation"""
        # Handle case where data is already in combined format
        if isinstance(data_input, dict) and 'document_condition' in data_input:
            combined = data_input
        else:
            # Handle both old format (document_data, vision_data) and new format (list of results)
            if isinstance(data_input, (list, tuple)) and len(data_input) == 2:
                # Old format: (document_data, vision_data)
                document_data, vision_data = data_input
                combined = self._combine_data_sources(document_data, vision_data)
            elif isinstance(data_input, list):
                # New format: list of results from document and vision analysis
                document_results = [r for r in data_input if r.get('document_type') in ['PDF', 'DOCX', 'TEXT']]
                vision_results = [r for r in data_input if r.get('document_type') == 'IMAGE']
                
                # Combine document results
                document_data = self._combine_document_results(document_results)
                vision_data = self._combine_vision_results(vision_results)
                combined = self._combine_data_sources(document_data, vision_data)
            else:
                # Single result format
                if data_input.get('document_type') in ['PDF', 'DOCX', 'TEXT']:
                    document_data = data_input
                    vision_data = {'property_condition': {'condition_score': 0.5}, 'hazards_detected': [], 'overall_score': 0.5}
                else:
                    document_data = {'property_condition': 'Unknown', 'hazards': [], 'risk_factors': []}
                    vision_data = data_input
                combined = self._combine_data_sources(document_data, vision_data)
        
        # Add data quality indicators
        combined['data_quality'] = {
            'completeness': self._calculate_data_completeness(combined),
            'consistency': self._check_data_consistency(combined),
            'reliability': self._assess_data_reliability(combined)
        }
        
        return combined
    
    def _perform_statistical_analysis(self, data: Dict) -> Dict:
        """Perform comprehensive statistical analysis"""
        analysis = {
            'descriptive_stats': {},
            'percentile_analysis': {},
            'z_scores': {},
            'statistical_tests': {}
        }
        
        # Analyze numerical fields
        numerical_fields = ['property_value', 'square_footage', 'year_built']
        
        for field in numerical_fields:
            value = data.get(field)
            if value is not None and isinstance(value, (int, float)):
                # Descriptive statistics
                historical_data = self.historical_patterns.get(field, {})
                if historical_data:
                    mean = historical_data['mean']
                    std_dev = historical_data['std_dev']
                    
                    # Z-score calculation
                    z_score = (value - mean) / std_dev if std_dev > 0 else 0
                    
                    # Percentile calculation
                    percentile = self._calculate_percentile(value, historical_data)
                    
                    analysis['descriptive_stats'][field] = {
                        'value': value,
                        'mean': mean,
                        'std_dev': std_dev,
                        'z_score': z_score,
                        'percentile': percentile
                    }
                    
                    analysis['z_scores'][field] = z_score
                    analysis['percentile_analysis'][field] = percentile
        
        # Statistical significance tests
        analysis['statistical_tests'] = self._perform_statistical_tests(data)
        
        return analysis
    
    def _calculate_percentile(self, value: float, historical_data: Dict) -> float:
        """Calculate percentile rank based on historical data"""
        percentiles = historical_data.get('percentiles', [])
        percentile_values = historical_data.get('percentile_values', [])
        
        if not percentiles or not percentile_values:
            return 50.0  # Default to median
        
        # Find the appropriate percentile
        for i, p_val in enumerate(percentile_values):
            if value <= p_val:
                return percentiles[i]
        
        return 95.0  # If above all percentiles
    
    def _perform_statistical_tests(self, data: Dict) -> Dict:
        """Perform statistical tests for data validation"""
        tests = {}
        
        # Test for normality (if we have enough data points)
        numerical_values = []
        for field in ['property_value', 'square_footage', 'year_built']:
            value = data.get(field)
            if value is not None and isinstance(value, (int, float)):
                numerical_values.append(value)
        
        if len(numerical_values) >= 3:
            try:
                # Shapiro-Wilk test for normality
                statistic, p_value = stats.shapiro(numerical_values)
                tests['normality_test'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
            except:
                tests['normality_test'] = {'error': 'Insufficient data for normality test'}
        
        return tests
    
    def _calculate_correlations(self, data: Dict) -> Dict:
        """Calculate correlations between different risk factors"""
        correlations = {}
        
        # Create correlation matrix for numerical variables
        numerical_data = {}
        for field in ['property_value', 'square_footage', 'year_built']:
            value = data.get(field)
            if value is not None and isinstance(value, (int, float)):
                numerical_data[field] = value
        
        if len(numerical_data) >= 2:
            # Calculate pairwise correlations
            fields = list(numerical_data.keys())
            for i, field1 in enumerate(fields):
                for j, field2 in enumerate(fields[i+1:], i+1):
                    # For single values, we'll use historical correlations
                    correlation = self._get_historical_correlation(field1, field2)
                    correlations[f"{field1}_vs_{field2}"] = {
                        'correlation': correlation,
                        'significance': abs(correlation) > self.statistical_params['correlation_threshold']
                    }
        
        # Analyze hazard correlations
        hazard_correlations = self._analyze_hazard_correlations(data)
        correlations['hazard_correlations'] = hazard_correlations
        
        return correlations
    
    def _get_historical_correlation(self, field1: str, field2: str) -> float:
        """Get historical correlation between fields"""
        # Predefined correlations based on typical property data
        correlation_matrix = {
            ('property_value', 'square_footage'): 0.7,
            ('property_value', 'year_built'): -0.3,
            ('square_footage', 'year_built'): -0.2
        }
        
        return correlation_matrix.get((field1, field2), 0.0)
    
    def _analyze_hazard_correlations(self, data: Dict) -> Dict:
        """Analyze correlations between different hazard types"""
        hazard_correlations = {}
        
        # Get all hazards
        all_hazards = []
        all_hazards.extend(data.get('document_hazards', []))
        all_hazards.extend([h.get('type') for h in data.get('vision_hazards', [])])
        
        # Check for common hazard combinations
        hazard_combinations = {
            ('water_damage', 'mold'): 0.8,
            ('structural', 'foundation'): 0.9,
            ('electrical', 'fire_hazard'): 0.7,
            ('water_damage', 'structural'): 0.6,
            ('maintenance', 'electrical'): 0.5
        }
        
        for hazard1 in all_hazards:
            for hazard2 in all_hazards:
                if hazard1 != hazard2:
                    key = tuple(sorted([hazard1.lower(), hazard2.lower()]))
                    if key in hazard_combinations:
                        hazard_correlations[f"{hazard1}_and_{hazard2}"] = {
                            'correlation': hazard_combinations[key],
                            'significance': True
                        }
        
        return hazard_correlations
    
    def _detect_outliers(self, data: Dict) -> Dict:
        """Detect outliers in the data"""
        outliers = {}
        
        # Check for outliers in numerical fields
        for field in ['property_value', 'square_footage', 'year_built']:
            value = data.get(field)
            if value is not None and isinstance(value, (int, float)):
                historical_data = self.historical_patterns.get(field, {})
                if historical_data:
                    mean = historical_data['mean']
                    std_dev = historical_data['std_dev']
                    
                    z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0
                    is_outlier = z_score > self.statistical_params['outlier_threshold']
                    
                    outliers[field] = {
                        'value': value,
                        'z_score': z_score,
                        'is_outlier': is_outlier,
                        'severity': 'high' if z_score > 3 else 'medium' if z_score > 2 else 'low'
                    }
        
        return outliers
    
    def _assess_property_condition_enhanced(self, data: Dict) -> Dict:
        """Enhanced property condition assessment with statistical analysis"""
        # Get condition from document analysis
        doc_condition = data.get('document_condition', 'Unknown')
        if doc_condition is None:
            doc_condition = 'Unknown'
        doc_condition = doc_condition.lower()
        
        # Get condition from vision analysis
        vision_score = data.get('vision_condition_score', 0.5)
        
        # Map vision score to condition category
        if vision_score >= 0.8:
            vision_condition = 'excellent'
        elif vision_score >= 0.6:
            vision_condition = 'good'
        elif vision_score >= 0.4:
            vision_condition = 'fair'
        else:
            vision_condition = 'poor'
        
        # Combine assessments (weighted average)
        doc_weight = 0.6
        vision_weight = 0.4
        
        doc_score = self.guidelines['property_condition'].get(doc_condition, 
                   self.guidelines['property_condition']['unknown'])['score']
        vision_score_normalized = self.guidelines['property_condition'][vision_condition]['score']
        
        combined_score = (doc_score * doc_weight + vision_score_normalized * vision_weight)
        
        # Determine final condition category
        if combined_score >= 0.8:
            final_condition = 'excellent'
        elif combined_score >= 0.6:
            final_condition = 'good'
        elif combined_score >= 0.4:
            final_condition = 'fair'
        else:
            final_condition = 'poor'
        
        risk_multiplier = self.guidelines['property_condition'][final_condition]['risk_multiplier']
        
        # Add statistical enhancements
        condition_guideline = self.guidelines['property_condition'].get(
            final_condition, 
            self.guidelines['property_condition']['unknown']
        )
        
        confidence_interval = condition_guideline.get('confidence_interval', 0.1)
        
        # Statistical significance
        statistical_significance = self._calculate_statistical_significance(
            combined_score, condition_guideline.get('std_dev', 0.2)
        )
        
        return {
            'condition': final_condition,
            'score': combined_score,
            'risk_multiplier': risk_multiplier,
            'risk_score': 1 - combined_score,
            'confidence': 0.8 if doc_condition != 'unknown' else 0.6,
            'confidence_interval': confidence_interval,
            'statistical_significance': statistical_significance,
            'percentile_rank': self._calculate_condition_percentile(combined_score)
        }
    
    def _calculate_statistical_significance(self, score: float, std_dev: float) -> Dict:
        """Calculate statistical significance of a score"""
        if std_dev == 0:
            return {'significant': True, 'p_value': 0.0}
        
        # Calculate z-score relative to expected distribution
        z_score = abs(score - 0.5) / std_dev
        p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
        
        return {
            'significant': p_value < (1 - self.statistical_params['confidence_level']),
            'p_value': p_value,
            'z_score': z_score
        }
    
    def _calculate_condition_percentile(self, score: float) -> float:
        """Calculate percentile rank for condition score"""
        # Assume condition scores follow a normal distribution
        # with mean 0.6 and std_dev 0.2
        mean = 0.6
        std_dev = 0.2
        
        z_score = (score - mean) / std_dev
        percentile = stats.norm.cdf(z_score) * 100
        
        return percentile
    
    def _calculate_overall_risk_enhanced(self, risk_components: Dict, statistical_analysis: Dict) -> Dict:
        """Calculate overall risk with confidence intervals and statistical significance"""
        # Calculate weighted risk score
        total_risk = 0
        total_weight = 0
        weighted_variance = 0
        
        for component_name, component_data in risk_components.items():
            weight_info = self.risk_weights.get(component_name.replace('_risk', ''), {'weight': 0.1, 'confidence': 0.5})
            weight = weight_info['weight']
            confidence = weight_info['confidence']
            
            risk_score = component_data.get('risk_score', 0)
            
            total_risk += risk_score * weight
            total_weight += weight
            
            # Calculate variance contribution
            confidence_interval = component_data.get('confidence_interval', 0.1)
            weighted_variance += (weight * confidence_interval) ** 2
        
        # Calculate overall risk score
        if total_weight > 0:
            overall_risk_score = total_risk / total_weight
        else:
            overall_risk_score = 0.5
        
        # Calculate confidence interval
        overall_std_error = math.sqrt(weighted_variance)
        confidence_interval = overall_std_error * 1.96  # 95% confidence interval
        
        # Statistical significance
        statistical_significance = self._calculate_statistical_significance(
            overall_risk_score, overall_std_error
        )
        
        return {
            'score': overall_risk_score,
            'confidence_interval': confidence_interval,
            'statistical_significance': statistical_significance,
            'weighted_variance': weighted_variance
        }
    
    def _categorize_risk_enhanced(self, overall_risk: Dict) -> Dict:
        """Enhanced risk categorization with confidence intervals"""
        score = overall_risk['score']
        confidence_interval = overall_risk['confidence_interval']
        
        # Determine category with confidence
        if score <= self.risk_thresholds['low']['threshold']:
            category = 'Low Risk'
            threshold_confidence = self.risk_thresholds['low']['confidence_interval']
        elif score <= self.risk_thresholds['medium']['threshold']:
            category = 'Medium Risk'
            threshold_confidence = self.risk_thresholds['medium']['confidence_interval']
        elif score <= self.risk_thresholds['high']['threshold']:
            category = 'High Risk'
            threshold_confidence = self.risk_thresholds['high']['confidence_interval']
        else:
            category = 'Very High Risk'
            threshold_confidence = 0.15
        
        # Check if categorization is statistically significant
        categorization_confidence = 1 - (confidence_interval / threshold_confidence)
        categorization_confidence = max(0, min(1, categorization_confidence))
        
        return {
            'category': category,
            'confidence': categorization_confidence,
            'threshold_used': threshold_confidence
        }
    
    def _generate_recommendations_enhanced(self, risk_components: Dict, overall_risk: Dict, 
                                         statistical_analysis: Dict, outlier_analysis: Dict) -> List[str]:
        """Generate enhanced recommendations based on statistical analysis"""
        recommendations = []
        
        # Base recommendations
        base_recommendations = self._generate_base_recommendations(risk_components, overall_risk['score'])
        recommendations.extend(base_recommendations)
        
        # Statistical analysis recommendations
        for field, outlier_info in outlier_analysis.items():
            if outlier_info['is_outlier']:
                severity = outlier_info['severity']
                recommendations.append(f"Statistical outlier detected in {field}: {severity} severity")
                if severity == 'high':
                    recommendations.append(f"Verify {field} data accuracy - value significantly outside normal range")
        
        # Correlation-based recommendations
        if 'correlation_analysis' in statistical_analysis:
            correlations = statistical_analysis['correlation_analysis']
            for corr_key, corr_info in correlations.items():
                if isinstance(corr_info, dict) and corr_info.get('significance'):
                    recommendations.append(f"Strong correlation detected: {corr_key}")
        
        # Data quality recommendations
        if statistical_analysis.get('data_quality_score', 1.0) < 0.7:
            recommendations.append("Data quality concerns detected - consider additional verification")
        
        return recommendations
    
    def _generate_base_recommendations(self, risk_components: Dict, overall_risk: float) -> List[str]:
        """Generate base recommendations based on risk assessment"""
        recommendations = []
        
        # Overall risk recommendations
        if overall_risk > 0.8:
            recommendations.append("Immediate action required - property poses significant risk")
            recommendations.append("Consider professional inspection before proceeding")
        elif overall_risk > 0.6:
            recommendations.append("Additional due diligence recommended")
            recommendations.append("Consider property inspection")
        elif overall_risk > 0.4:
            recommendations.append("Standard underwriting procedures apply")
        else:
            recommendations.append("Property appears to be in good condition")
        
        # Component-specific recommendations
        hazard_risk = risk_components.get('hazard_risk', {})
        if hazard_risk.get('total_hazards', 0) > 0:
            recommendations.append(f"Address {hazard_risk['total_hazards']} identified hazards")
        
        age_risk = risk_components.get('age_risk', {})
        age_category = age_risk.get('age_category', 'unknown')
        if age_category in ['older', 'historic']:
            recommendations.append("Consider age-related maintenance requirements")
        
        maintenance_risk = risk_components.get('maintenance_risk', {})
        maintenance_issues = maintenance_risk.get('maintenance_issues', [])
        if maintenance_issues:
            recommendations.append("Address maintenance issues before approval")
        
        return recommendations
    
    def _calculate_confidence_enhanced(self, data: Dict, statistical_analysis: Dict) -> float:
        """Calculate enhanced confidence score"""
        base_confidence = self._calculate_base_confidence(data)
        
        # Adjust based on statistical analysis
        statistical_factors = []
        
        # Data completeness
        if 'descriptive_stats' in statistical_analysis:
            num_fields_analyzed = len(statistical_analysis['descriptive_stats'])
            statistical_factors.append(min(1.0, num_fields_analyzed / 3))
        
        # Outlier presence
        outlier_count = sum(1 for info in statistical_analysis.get('outlier_analysis', {}).values() 
                           if info.get('is_outlier', False))
        outlier_factor = max(0.5, 1.0 - (outlier_count * 0.1))
        statistical_factors.append(outlier_factor)
        
        # Statistical significance
        significant_tests = sum(1 for test in statistical_analysis.get('statistical_tests', {}).values()
                               if isinstance(test, dict) and test.get('significant', False))
        significance_factor = min(1.0, significant_tests / 2)
        statistical_factors.append(significance_factor)
        
        # Combine factors
        if statistical_factors:
            statistical_confidence = sum(statistical_factors) / len(statistical_factors)
            enhanced_confidence = (base_confidence * 0.7 + statistical_confidence * 0.3)
        else:
            enhanced_confidence = base_confidence
        
        return min(1.0, max(0.0, enhanced_confidence))
    
    def _calculate_base_confidence(self, data: Dict) -> float:
        """Calculate base confidence in the assessment"""
        confidence_factors = []
        
        # Document data quality
        if data.get('property_address'):
            confidence_factors.append(0.8)
        if data.get('year_built'):
            confidence_factors.append(0.9)
        if data.get('property_value'):
            confidence_factors.append(0.7)
        
        # Vision data quality
        image_quality = data.get('image_quality', 0.5)
        confidence_factors.append(image_quality)
        
        # Data completeness
        data_fields = ['property_address', 'property_value', 'square_footage', 
                      'bedrooms', 'bathrooms', 'year_built']
        filled_fields = sum(1 for field in data_fields if data.get(field) is not None)
        completeness = filled_fields / len(data_fields)
        confidence_factors.append(completeness)
        
        # Calculate average confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5
    
    def _assess_data_quality(self, data: Dict) -> Dict:
        """Assess overall data quality"""
        try:
            quality_metrics = {
                'completeness': self._calculate_data_completeness(data),
                'consistency': self._check_data_consistency(data),
                'reliability': self._assess_data_reliability(data),
                'validity': self._check_data_validity(data)
            }
            
            overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'overall_score': overall_quality,
                'metrics': quality_metrics,
                'grade': self._get_quality_grade(overall_quality)
            }
        except Exception as e:
            return {
                'overall_score': 0.5,
                'metrics': {},
                'grade': 'C',
                'error': str(e)
            }
    
    def _calculate_data_completeness(self, data: Dict) -> float:
        """Calculate data completeness score"""
        required_fields = ['property_address', 'property_value', 'square_footage', 
                          'bedrooms', 'bathrooms', 'year_built']
        
        filled_fields = sum(1 for field in required_fields if data.get(field) is not None)
        return filled_fields / len(required_fields)
    
    def _check_data_consistency(self, data: Dict) -> float:
        """Check data consistency"""
        consistency_score = 1.0
        
        # Check for logical inconsistencies
        if data.get('bedrooms') and data.get('square_footage'):
            bedrooms = data['bedrooms']
            sqft = data['square_footage']
            if bedrooms > 0 and sqft > 0:
                sqft_per_bedroom = sqft / bedrooms
                if sqft_per_bedroom < 200 or sqft_per_bedroom > 2000:
                    consistency_score -= 0.3
        
        if data.get('year_built') and data.get('property_value'):
            year_built = data['year_built']
            value = data['property_value']
            current_year = datetime.now().year
            
            if year_built > current_year:
                consistency_score -= 0.5
            elif year_built < 1800:
                consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _assess_data_reliability(self, data: Dict) -> float:
        """Assess data reliability"""
        reliability_score = 1.0
        
        # Check for multiple data sources
        has_document_data = any(data.get(field) for field in ['document_condition', 'document_hazards'])
        has_vision_data = any(data.get(field) for field in ['vision_condition_score', 'vision_hazards'])
        
        if has_document_data and has_vision_data:
            reliability_score += 0.2
        elif has_document_data or has_vision_data:
            reliability_score += 0.1
        
        # Check for detailed information
        if len(data.get('document_hazards', [])) > 0:
            reliability_score += 0.1
        
        if len(data.get('vision_hazards', [])) > 0:
            reliability_score += 0.1
        
        return min(1.0, reliability_score)
    
    def _check_data_validity(self, data: Dict) -> float:
        """Check data validity"""
        validity_score = 1.0
        
        # Check for reasonable values
        if data.get('property_value'):
            value = data['property_value']
            if value < 10000 or value > 10000000:  # Unreasonable property values
                validity_score -= 0.3
        
        if data.get('square_footage'):
            sqft = data['square_footage']
            if sqft < 100 or sqft > 20000:  # Unreasonable square footage
                validity_score -= 0.3
        
        if data.get('bedrooms'):
            bedrooms = data['bedrooms']
            if bedrooms < 0 or bedrooms > 20:  # Unreasonable bedroom count
                validity_score -= 0.3
        
        return max(0.0, validity_score)
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _get_default_assessment(self, error: str) -> Dict:
        """Return default assessment when errors occur"""
        return {
            'risk_score': 0.5,
            'risk_category': 'Medium Risk',
            'confidence_interval': 0.2,
            'statistical_significance': {'significant': False, 'p_value': 1.0},
            'risk_components': {
                'property_condition_risk': {'risk_score': 0.5, 'confidence': 0.5},
                'hazard_risk': {'risk_score': 0.5, 'confidence': 0.5},
                'age_risk': {'risk_score': 0.5, 'confidence': 0.5},
                'location_risk': {'risk_score': 0.5, 'confidence': 0.5},
                'maintenance_risk': {'risk_score': 0.5, 'confidence': 0.5}
            },
            'statistical_analysis': {},
            'correlation_analysis': {},
            'outlier_analysis': {},
            'recommendations': ['Standard underwriting procedures apply'],
            'guideline_compliance': {'compliance_score': 0.8, 'overall_compliant': True},
            'assessment_timestamp': datetime.now().isoformat(),
            'confidence_score': 0.5,
            'data_quality_score': {'overall_score': 0.5, 'grade': 'C'},
            'error': error
        }
    
    def _combine_data_sources(self, document_data: Dict, vision_data: Dict) -> Dict:
        """Combine and validate data from multiple sources"""
        # Handle case where data is already in the expected format
        if isinstance(document_data, dict) and 'document_condition' in document_data:
            # Data is already in combined format
            return document_data
        
        combined = {
            'property_address': document_data.get('property_address'),
            'property_value': document_data.get('property_value'),
            'square_footage': document_data.get('square_footage'),
            'bedrooms': document_data.get('bedrooms'),
            'bathrooms': document_data.get('bathrooms'),
            'year_built': document_data.get('year_built'),
            'document_condition': document_data.get('property_condition', 'Unknown'),
            'document_hazards': document_data.get('hazards', []),
            'document_risk_factors': document_data.get('risk_factors', []),
            'vision_condition_score': vision_data.get('property_condition', {}).get('condition_score', 0.5),
            'vision_hazards': vision_data.get('hazards_detected', []),
            'vision_overall_score': vision_data.get('overall_score', 0.5),
            'image_quality': vision_data.get('image_quality', {}).get('quality_score', 0.5)
        }
        
        return combined
    
    def _assess_hazard_risk_enhanced(self, data: Dict) -> Dict:
        """Enhanced hazard risk assessment"""
        # Combine hazards from both sources
        all_hazards = []
        
        # Document hazards
        for hazard in data.get('document_hazards', []):
            all_hazards.append({
                'type': hazard.lower(),
                'source': 'document',
                'severity': 'medium'  # Default severity
            })
        
        # Vision hazards
        for hazard in data.get('vision_hazards', []):
            all_hazards.append({
                'type': hazard['type'],
                'source': 'vision',
                'severity': self._categorize_hazard_severity(hazard['severity']),
                'confidence': hazard['confidence']
            })
        
        # Calculate hazard risk
        total_hazard_risk = 0
        hazard_details = []
        
        for hazard in all_hazards:
            hazard_type = hazard['type']
            severity = hazard['severity']
            
            if hazard_type in self.guidelines['hazards']:
                severity_key = f'severity_{severity}'
                if severity_key in self.guidelines['hazards'][hazard_type]:
                    risk_score = self.guidelines['hazards'][hazard_type][severity_key]['risk']
                else:
                    risk_score = 0.5  # Default
            else:
                risk_score = 0.5  # Default for unknown hazards
            
            total_hazard_risk += risk_score
            hazard_details.append({
                'type': hazard_type,
                'severity': severity,
                'risk_score': risk_score,
                'source': hazard['source']
            })
        
        # Normalize total risk
        if all_hazards:
            avg_hazard_risk = total_hazard_risk / len(all_hazards)
        else:
            avg_hazard_risk = 0
        
        return {
            'total_hazards': len(all_hazards),
            'hazard_details': hazard_details,
            'risk_score': avg_hazard_risk,
            'confidence': 0.7 if all_hazards else 0.5
        }
    
    def _categorize_hazard_severity(self, severity_score: float) -> str:
        """Categorize hazard severity based on score"""
        if severity_score >= 0.7:
            return 'high'
        elif severity_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _assess_age_risk_enhanced(self, data: Dict) -> Dict:
        """Enhanced age risk assessment"""
        year_built = data.get('year_built')
        current_year = datetime.now().year
        
        if year_built is None:
            return {
                'age': 'unknown',
                'risk_multiplier': 1.2,
                'risk_score': 0.6,
                'confidence': 0.3
            }
        
        age = current_year - year_built
        
        # Determine age category
        if age <= 5:
            age_category = 'new'
        elif age <= 15:
            age_category = 'recent'
        elif age <= 35:
            age_category = 'established'
        elif age <= 75:
            age_category = 'older'
        else:
            age_category = 'historic'
        
        risk_multiplier = self.guidelines['property_age'][age_category]['risk_multiplier']
        
        return {
            'age': age,
            'age_category': age_category,
            'risk_multiplier': risk_multiplier,
            'risk_score': (risk_multiplier - 0.6) / 1.0,  # Normalize to 0-1
            'confidence': 0.9
        }
    
    def _assess_location_risk_enhanced(self, data: Dict) -> Dict:
        """Enhanced location risk assessment"""
        address = data.get('property_address', '').lower()
        location_risks = []
        total_location_risk = 0
        
        # Check for common location risk indicators
        if 'flood' in address or 'river' in address or 'creek' in address:
            location_risks.append('flood_zone')
            total_location_risk += self.guidelines['location_risk']['flood_zone']['risk']
        
        if 'earthquake' in address or 'fault' in address:
            location_risks.append('earthquake_zone')
            total_location_risk += self.guidelines['location_risk']['earthquake_zone']['risk']
        
        if 'industrial' in address or 'factory' in address:
            location_risks.append('industrial_area')
            total_location_risk += self.guidelines['location_risk']['industrial_area']['risk']
        
        # Normalize risk
        if location_risks:
            avg_location_risk = total_location_risk / len(location_risks)
        else:
            avg_location_risk = 0.1  # Base location risk
        
        return {
            'location_risks': location_risks,
            'risk_score': avg_location_risk,
            'confidence': 0.6 if location_risks else 0.4
        }
    
    def _assess_maintenance_risk_enhanced(self, data: Dict) -> Dict:
        """Enhanced maintenance risk assessment"""
        maintenance_issues = []
        
        # From document analysis
        risk_factors = data.get('document_risk_factors', [])
        for factor in risk_factors:
            if 'maintenance' in factor.lower():
                maintenance_issues.append(factor)
        
        # From vision analysis
        vision_score = data.get('vision_overall_score', 0.5)
        if vision_score < 0.6:
            maintenance_issues.append('Poor visual condition')
        
        # Calculate maintenance risk
        if maintenance_issues:
            maintenance_risk = 0.6  # Moderate risk when issues detected
        else:
            maintenance_risk = 0.2  # Low risk when no issues detected
        
        return {
            'maintenance_issues': maintenance_issues,
            'risk_score': maintenance_risk,
            'confidence': 0.7 if maintenance_issues else 0.5
        }
    
    def _check_guideline_compliance_enhanced(self, data: Dict) -> Dict:
        """Enhanced guideline compliance check"""
        compliance_checks = {
            'property_condition_acceptable': True,
            'hazards_within_limits': True,
            'age_requirements_met': True,
            'location_acceptable': True,
            'maintenance_standards_met': True
        }
        
        # Check property condition
        condition = data.get('document_condition', 'Unknown').lower()
        if condition == 'poor':
            compliance_checks['property_condition_acceptable'] = False
        
        # Check hazards
        if len(data.get('document_hazards', [])) > 3:
            compliance_checks['hazards_within_limits'] = False
        
        # Check age
        year_built = data.get('year_built')
        if year_built and (datetime.now().year - year_built) > 100:
            compliance_checks['age_requirements_met'] = False
        
        # Calculate overall compliance
        compliant_checks = sum(compliance_checks.values())
        total_checks = len(compliance_checks)
        compliance_score = compliant_checks / total_checks
        
        return {
            'checks': compliance_checks,
            'compliance_score': compliance_score,
            'overall_compliant': compliance_score >= 0.8
        }
    
    def _combine_document_results(self, document_results: List[Dict]) -> Dict:
        """Combine multiple document analysis results"""
        if not document_results:
            return {'property_condition': 'Unknown', 'hazards': [], 'risk_factors': []}
        
        # Combine text from all documents
        combined_text = ""
        all_hazards = []
        all_risk_factors = []
        
        for result in document_results:
            combined_text += result.get('raw_text', '') + " "
            all_hazards.extend(result.get('hazards', []))
            all_risk_factors.extend(result.get('risk_factors', []))
        
        # Use the first result as base and update with combined data
        base_result = document_results[0].copy()
        base_result['raw_text'] = combined_text.strip()
        base_result['hazards'] = list(set(all_hazards))  # Remove duplicates
        base_result['risk_factors'] = list(set(all_risk_factors))  # Remove duplicates
        
        return base_result
    
    def _combine_vision_results(self, vision_results: List[Dict]) -> Dict:
        """Combine multiple vision analysis results"""
        if not vision_results:
            return {'property_condition': {'condition_score': 0.5}, 'hazards_detected': [], 'overall_score': 0.5}
        
        # Average the scores and combine hazards
        total_condition_score = 0
        total_overall_score = 0
        all_hazards = []
        
        for result in vision_results:
            condition_score = result.get('property_condition', {}).get('condition_score', 0.5)
            overall_score = result.get('overall_score', 0.5)
            hazards = result.get('hazards_detected', [])
            
            total_condition_score += condition_score
            total_overall_score += overall_score
            all_hazards.extend(hazards)
        
        # Calculate averages
        avg_condition_score = total_condition_score / len(vision_results)
        avg_overall_score = total_overall_score / len(vision_results)
        
        # Use the first result as base and update with combined data
        base_result = vision_results[0].copy()
        base_result['property_condition']['condition_score'] = avg_condition_score
        base_result['overall_score'] = avg_overall_score
        base_result['hazards_detected'] = all_hazards
        
        return base_result 