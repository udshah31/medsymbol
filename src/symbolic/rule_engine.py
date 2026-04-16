"""
Enhanced Symbolic Rule Engine
=============================

Advanced rule system with:
- Rule composition and validation
- Automatic rule discovery from training data
- Conflict detection and resolution
- Interpretable rule evaluation
- Performance tracking
"""

from typing import Dict, List, Tuple, Callable, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict


class RuleType(Enum):
    """Types of medical rules."""
    INCLUSION = "inclusion"  # Rule that increases belief
    EXCLUSION = "exclusion"  # Rule that decreases belief
    CONDITIONAL = "conditional"  # If X then Y
    NECESSITY = "necessity"  # X required for Y
    SUFFICIENCY = "sufficiency"  # X sufficient for Y


@dataclass
class Rule:
    """A single medical rule with metadata."""
    
    id: str
    disease: str
    rule_type: RuleType
    condition: Callable
    weight: float = 1.0  # Strength of rule
    confidence: float = 0.0  # Empirical confidence from data
    specificity: float = 0.0  # How specific is this rule
    sensitivity: float = 0.0  # How sensitive is this rule
    source: str = "manual"  # manual, learned, clinical_guideline
    evidence_count: int = 0  # How many times has this been validated
    exceptions: List[str] = field(default_factory=list)  # Known exceptions
    description: str = ""
    
    def evaluate(self, patient_data: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate rule on patient data.
        
        Returns:
            (matched, score, explanation)
        """
        try:
            matched = self.condition(patient_data)
            score = self.weight * self.confidence if matched else 0.0
            explanation = f"Rule '{self.description}' {'matched' if matched else 'not matched'}"
            return matched, score, explanation
        except Exception as e:
            return False, 0.0, f"Rule evaluation error: {str(e)}"


class RuleConflict:
    """Detected conflict between rules."""
    
    def __init__(self, rule1: Rule, rule2: Rule, conflict_type: str):
        self.rule1 = rule1
        self.rule2 = rule2
        self.conflict_type = conflict_type  # contradiction, redundancy, etc.
        self.severity = "high" if conflict_type == "contradiction" else "medium"


class RuleEngine:
    """
    Sophisticated medical rule engine with advanced features.
    """
    
    def __init__(self):
        self.rules: Dict[str, List[Rule]] = defaultdict(list)
        self.rule_history: List[Rule] = []
        self.conflicts: List[RuleConflict] = []
        self.learned_rules: List[Rule] = []
    
    def add_rule(self, disease: str, rule: Rule) -> bool:
        """Add a rule to the engine with validation."""
        # Validate rule doesn't conflict
        conflicts = self._detect_conflicts(rule)
        if conflicts:
            self.conflicts.extend(conflicts)
            # Log but don't reject
        
        self.rules[disease].append(rule)
        self.rule_history.append(rule)
        return True
    
    def evaluate_disease(self, disease: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive rule evaluation for a disease.
        
        Returns:
            {
                'disease': str,
                'overall_score': float,
                'matched_rules': List[Rule],
                'fired_rules': List[Dict],
                'reasoning': str,
                'confidence': float,
                'conflicts': List[str]
            }
        """
        disease_rules = self.rules.get(disease, [])
        
        if not disease_rules:
            return {
                'disease': disease,
                'overall_score': 0.0,
                'matched_rules': [],
                'fired_rules': [],
                'reasoning': f'No rules defined for {disease}',
                'confidence': 0.0,
                'conflicts': []
            }
        
        matched_rules = []
        fired_rules = []
        total_score = 0.0
        conflict_explanations = []
        
        # Evaluate all rules
        for rule in disease_rules:
            matched, score, explanation = rule.evaluate(patient_data)
            
            if matched:
                matched_rules.append(rule)
                fired_rules.append({
                    'id': rule.id,
                    'description': rule.description,
                    'score': score,
                    'confidence': rule.confidence,
                    'type': rule.rule_type.value
                })
                total_score += score
        
        # Detect fired conflicts
        for conflict in self._detect_conflicts_in_fired_rules(matched_rules):
            conflict_explanations.append(
                f"{conflict.rule1.description} conflicts with {conflict.rule2.description}"
            )
        
        # Calculate overall confidence
        max_score = sum(r.weight for r in disease_rules)
        overall_confidence = total_score / max_score if max_score > 0 else 0.0
        
        return {
            'disease': disease,
            'overall_score': min(total_score, 1.0),
            'matched_rules': matched_rules,
            'fired_rules': fired_rules,
            'reasoning': self._generate_explanation(fired_rules),
            'confidence': overall_confidence,
            'conflicts': conflict_explanations,
            'rule_count': len(disease_rules),
            'match_rate': len(matched_rules) / len(disease_rules) if disease_rules else 0.0
        }
    
    def learn_rules_from_data(self, training_data: List[Dict], 
                             disease: str, min_support: float = 0.1) -> List[Rule]:
        """
        Automatically discover rules from training data using frequent pattern mining.
        
        Args:
            training_data: List of labeled patient examples
            disease: Target disease
            min_support: Minimum support threshold
        
        Returns:
            List of discovered rules
        """
        discovered = []
        
        # Simple frequent pattern mining
        feature_frequencies = defaultdict(int)
        disease_occurrences = 0
        
        for example in training_data:
            if example.get('disease') == disease:
                disease_occurrences += 1
                
                # Extract boolean features
                for feature, value in example.items():
                    if isinstance(value, bool) or isinstance(value, (int, float)):
                        feature_frequencies[f"{feature}={value}"] += 1
        
        # Generate rules from frequent patterns
        for feature_pattern, count in feature_frequencies.items():
            support = count / len(training_data) if training_data else 0
            confidence = count / disease_occurrences if disease_occurrences > 0 else 0
            
            if support >= min_support:
                rule_id = f"learned_{disease}_{len(discovered)}"
                
                # Create rule from pattern
                def make_rule_condition(pattern: str):
                    def condition(data: Dict) -> bool:
                        feature, value_str = pattern.split('=')
                        if feature not in data:
                            return False
                        try:
                            return str(data[feature]) == value_str
                        except:
                            return False
                    return condition
                
                rule = Rule(
                    id=rule_id,
                    disease=disease,
                    rule_type=RuleType.INCLUSION,
                    condition=make_rule_condition(feature_pattern),
                    weight=support,
                    confidence=confidence,
                    source="learned",
                    description=f"Pattern {feature_pattern} (support={support:.2%}, confidence={confidence:.2%})"
                )
                
                discovered.append(rule)
        
        self.learned_rules.extend(discovered)
        return discovered
    
    def _detect_conflicts(self, new_rule: Rule) -> List[RuleConflict]:
        """Detect conflicts with existing rules."""
        conflicts = []
        disease_rules = self.rules.get(new_rule.disease, [])
        
        for existing in disease_rules:
            # Check for contradiction
            if (existing.rule_type == RuleType.INCLUSION and 
                new_rule.rule_type == RuleType.EXCLUSION):
                conflicts.append(RuleConflict(existing, new_rule, "contradiction"))
            
            # Check for redundancy
            if (existing.description == new_rule.description):
                conflicts.append(RuleConflict(existing, new_rule, "redundancy"))
        
        return conflicts
    
    def _detect_conflicts_in_fired_rules(self, fired_rules: List[Rule]) -> List[RuleConflict]:
        """Detect conflicts among fired rules."""
        conflicts = []
        
        for i, rule1 in enumerate(fired_rules):
            for rule2 in fired_rules[i+1:]:
                if (rule1.rule_type == RuleType.INCLUSION and 
                    rule2.rule_type == RuleType.EXCLUSION):
                    conflicts.append(RuleConflict(rule1, rule2, "contradiction"))
        
        return conflicts
    
    def _generate_explanation(self, fired_rules: List[Dict]) -> str:
        """Generate human-readable explanation from fired rules."""
        if not fired_rules:
            return "No rules matched."
        
        explanations = [f"• {r['description']}" for r in fired_rules]
        return "Matched rules: " + "\n".join(explanations)
    
    def get_rules_for_disease(self, disease: str) -> List[Rule]:
        """Get all rules for a disease."""
        return self.rules.get(disease, [])
    
    def rule_statistics(self, disease: str = None) -> Dict[str, Any]:
        """Get statistics about rules."""
        if disease:
            rules = self.rules.get(disease, [])
            return {
                'disease': disease,
                'total_rules': len(rules),
                'avg_confidence': sum(r.confidence for r in rules) / len(rules) if rules else 0,
                'avg_weight': sum(r.weight for r in rules) / len(rules) if rules else 0,
                'manual_rules': len([r for r in rules if r.source == 'manual']),
                'learned_rules': len([r for r in rules if r.source == 'learned']),
            }
        else:
            all_rules = [r for rules in self.rules.values() for r in rules]
            return {
                'total_rules': len(all_rules),
                'total_diseases': len(self.rules),
                'avg_rules_per_disease': len(all_rules) / len(self.rules) if self.rules else 0,
                'detected_conflicts': len(self.conflicts),
                'learned_rules': len(self.learned_rules),
            }


class RuleValidator:
    """Validate rules against clinical evidence."""
    
    @staticmethod
    def validate_rule_against_data(rule: Rule, test_data: List[Dict], 
                                   disease: str) -> Dict[str, float]:
        """
        Validate rule performance on test data.
        
        Returns:
            {
                'sensitivity': TP / (TP + FN),
                'specificity': TN / (TN + FP),
                'precision': TP / (TP + FP),
                'f1': harmonic mean,
                'accuracy': (TP + TN) / total
            }
        """
        tp = fp = tn = fn = 0
        
        for example in test_data:
            is_disease = example.get('disease') == disease
            matched, _, _ = rule.evaluate(example)
            
            if matched and is_disease:
                tp += 1
            elif matched and not is_disease:
                fp += 1
            elif not matched and is_disease:
                fn += 1
            else:
                tn += 1
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }


if __name__ == "__main__":
    engine = RuleEngine()
    
    # Example: Add a pneumonia rule
    def pneumonia_condition(data: Dict) -> bool:
        return (data.get('fever', False) and 
                data.get('cough', False) and 
                data.get('consolidation_on_xray', False))
    
    pneumonia_rule = Rule(
        id="pneumonia_classic",
        disease="pneumonia",
        rule_type=RuleType.INCLUSION,
        condition=pneumonia_condition,
        weight=0.8,
        confidence=0.85,
        source="clinical_guideline",
        description="Classic presentation: fever + cough + consolidation"
    )
    
    engine.add_rule("pneumonia", pneumonia_rule)
    
    # Evaluate
    patient = {
        'fever': True,
        'cough': True,
        'consolidation_on_xray': True,
        'age': 45
    }
    
    result = engine.evaluate_disease("pneumonia", patient)
    print(json.dumps(result, indent=2, default=str))
