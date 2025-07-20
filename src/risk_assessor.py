"""
Risk Assessment Engine
Evaluates property risk and applies underwriting guidelines
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


class RiskAssessor:
    """Assesses property risk based on multiple data sources"""
    
    def __init__(self):
        # Load underwriting guidelines
        self.guidelines = self._load_guidelines()
        
        # Risk weights for different factors
        self.risk_weights = {
            'property_condition': 0.25,
            'hazards': 0.30,
            'property_age': 0.15,
            'location_risk': 0.10,
            'maintenance_issues': 0.20
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def _load_guidelines(self) -> Dict:
        """Load underwriting guidelines"""
        return {
            'property_condition': {
                'excellent': {'score': 1.0, 'risk_multiplier': 0.5},
                'good': {'score': 0.8, 'risk_multiplier': 0.7},
                'fair': {'score': 0.6, 'risk_multiplier': 1.0},
                'poor': {'score': 0.3, 'risk_multiplier': 1.5},
                'unknown': {'score': 0.5, 'risk_multiplier': 1.2}
            },
            'hazards': {
                'water_damage': {'severity_high': 0.8, 'severity_medium': 0.5, 'severity_low': 0.2},
                'structural': {'severity_high': 0.9, 'severity_medium': 0.6, 'severity_low': 0.3},
                'electrical': {'severity_high': 0.7, 'severity_medium': 0.4, 'severity_low': 0.2},
                'fire_hazard': {'severity_high': 0.9, 'severity_medium': 0.6, 'severity_low': 0.3},
                'maintenance': {'severity_high': 0.6, 'severity_medium': 0.4, 'severity_low': 0.2}
            },
            'property_age': {
                'new': {'years': 0, 'risk_multiplier': 0.6},
                'recent': {'years': 10, 'risk_multiplier': 0.8},
                'established': {'years': 25, 'risk_multiplier': 1.0},
                'older': {'years': 50, 'risk_multiplier': 1.3},
                'historic': {'years': 100, 'risk_multiplier': 1.6}
            },
            'location_risk': {
                'flood_zone': 0.8,
                'earthquake_zone': 0.6,
                'hurricane_zone': 0.7,
                'high_crime': 0.5,
                'industrial_area': 0.4
            }
        }
    
    def assess_risk(self, data_input) -> Dict:
        """Main method to assess property risk"""
        try:
            # Handle both old format (document_data, vision_data) and new format (list of results)
            if isinstance(data_input, (list, tuple)) and len(data_input) == 2:
                # Old format: (document_data, vision_data)
                document_data, vision_data = data_input
                combined_data = self._combine_data_sources(document_data, vision_data)
            elif isinstance(data_input, list):
                # New format: list of results from document and vision analysis
                document_results = [r for r in data_input if r.get('document_type') in ['PDF', 'DOCX', 'TEXT']]
                vision_results = [r for r in data_input if r.get('document_type') == 'IMAGE']
                
                # Combine document results
                document_data = self._combine_document_results(document_results)
                vision_data = self._combine_vision_results(vision_results)
                combined_data = self._combine_data_sources(document_data, vision_data)
            else:
                # Single result format
                if data_input.get('document_type') in ['PDF', 'DOCX', 'TEXT']:
                    document_data = data_input
                    vision_data = {'property_condition': {'condition_score': 0.5}, 'hazards_detected': [], 'overall_score': 0.5}
                else:
                    document_data = {'property_condition': 'Unknown', 'hazards': [], 'risk_factors': []}
                    vision_data = data_input
                combined_data = self._combine_data_sources(document_data, vision_data)
            
            # Calculate individual risk components
            risk_components = {
                'property_condition_risk': self._assess_property_condition_risk(combined_data),
                'hazard_risk': self._assess_hazard_risk(combined_data),
                'age_risk': self._assess_age_risk(combined_data),
                'location_risk': self._assess_location_risk(combined_data),
                'maintenance_risk': self._assess_maintenance_risk(combined_data)
            }
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk(risk_components)
            
            # Determine risk category
            risk_category = self._categorize_risk(overall_risk)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_components, overall_risk)
            
            # Create comprehensive assessment
            assessment = {
                'risk_score': overall_risk,
                'risk_category': risk_category,
                'risk_components': risk_components,
                'recommendations': recommendations,
                'guideline_compliance': self._check_guideline_compliance(combined_data),
                'assessment_timestamp': datetime.now().isoformat(),
                'confidence_score': self._calculate_confidence(combined_data)
            }
            
            return assessment
            
        except Exception as e:
            # Return a safe default assessment if anything goes wrong
            print(f"Error in risk assessment: {str(e)}")
            return {
                'risk_score': 0.5,
                'risk_category': 'Medium Risk',
                'risk_components': {
                    'property_condition_risk': {'risk_score': 0.5, 'confidence': 0.5},
                    'hazard_risk': {'risk_score': 0.5, 'confidence': 0.5},
                    'age_risk': {'risk_score': 0.5, 'confidence': 0.5},
                    'location_risk': {'risk_score': 0.5, 'confidence': 0.5},
                    'maintenance_risk': {'risk_score': 0.5, 'confidence': 0.5}
                },
                'recommendations': ['Standard underwriting procedures apply'],
                'guideline_compliance': {'compliance_score': 0.8, 'overall_compliant': True},
                'assessment_timestamp': datetime.now().isoformat(),
                'confidence_score': 0.5,
                'error': str(e)
            }
    
    def _combine_data_sources(self, document_data: Dict, vision_data: Dict) -> Dict:
        """Combine and validate data from multiple sources"""
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
    
    def _assess_property_condition_risk(self, data: Dict) -> Dict:
        """Assess risk based on property condition"""
        # Get condition from document analysis
        doc_condition = data.get('document_condition', 'Unknown').lower()
        
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
        
        return {
            'condition': final_condition,
            'score': combined_score,
            'risk_multiplier': risk_multiplier,
            'risk_score': 1 - combined_score,
            'confidence': 0.8 if doc_condition != 'Unknown' else 0.6
        }
    
    def _assess_hazard_risk(self, data: Dict) -> Dict:
        """Assess risk based on detected hazards"""
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
                    risk_score = self.guidelines['hazards'][hazard_type][severity_key]
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
    
    def _assess_age_risk(self, data: Dict) -> Dict:
        """Assess risk based on property age"""
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
    
    def _assess_location_risk(self, data: Dict) -> Dict:
        """Assess risk based on location factors"""
        # This would typically integrate with external data sources
        # For now, we'll use a simplified approach based on address patterns
        
        address = data.get('property_address', '').lower()
        location_risks = []
        total_location_risk = 0
        
        # Check for common location risk indicators
        if 'flood' in address or 'river' in address or 'creek' in address:
            location_risks.append('flood_zone')
            total_location_risk += self.guidelines['location_risk']['flood_zone']
        
        if 'earthquake' in address or 'fault' in address:
            location_risks.append('earthquake_zone')
            total_location_risk += self.guidelines['location_risk']['earthquake_zone']
        
        if 'industrial' in address or 'factory' in address:
            location_risks.append('industrial_area')
            total_location_risk += self.guidelines['location_risk']['industrial_area']
        
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
    
    def _assess_maintenance_risk(self, data: Dict) -> Dict:
        """Assess risk based on maintenance issues"""
        # Combine maintenance indicators from multiple sources
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
    
    def _calculate_overall_risk(self, risk_components: Dict) -> float:
        """Calculate overall risk score using weighted components"""
        total_risk = 0
        total_weight = 0
        
        for component_name, component_data in risk_components.items():
            weight = self.risk_weights.get(component_name.replace('_risk', ''), 0.1)
            risk_score = component_data.get('risk_score', 0)
            
            total_risk += risk_score * weight
            total_weight += weight
        
        # Normalize to 0-1 range
        if total_weight > 0:
            overall_risk = total_risk / total_weight
        else:
            overall_risk = 0.5
        
        return min(1.0, max(0.0, overall_risk))
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk based on score"""
        if risk_score <= self.risk_thresholds['low']:
            return 'Low Risk'
        elif risk_score <= self.risk_thresholds['medium']:
            return 'Medium Risk'
        elif risk_score <= self.risk_thresholds['high']:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def _generate_recommendations(self, risk_components: Dict, overall_risk: float) -> List[str]:
        """Generate recommendations based on risk assessment"""
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
    
    def _check_guideline_compliance(self, data: Dict) -> Dict:
        """Check compliance with underwriting guidelines"""
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
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence in the assessment"""
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