"""
점수 산출 모듈 (scorer.py)

기능:
- 비교 분석 결과를 종합하여 최종 점수 계산
- 가중치 기반 점수 산출(SCORING_WEIGHTS에서 설정 가능)

입력: 비교 분석 결과 딕셔너리들
출력: 최종 점수 및 세부 점수 JSON
"""

import json
from pathlib import Path

# 점수 가중치 설정
SCORING_WEIGHTS = {
    "pronunciation": 50,    # 발음 정확도 (가장 중요)
    "timing": 25,          # 타이밍 정확도  
    "pitch": 25            # 피치 유사도
}

def calculate_final_score(comparison_results: dict) -> dict:
    """
    비교 분석 결과를 바탕으로 최종 점수 계산
    
    Args:
        comparison_results: 각종 비교 분석 결과
    
    Returns:
        최종 점수 및 세부 점수 딕셔너리
    """
    print("🏆 최종 점수 계산 중...")
    
    try:
        # 각 항목별 점수 추출
        overall_scores = comparison_results.get('overall_scores', {})
        
        pronunciation_score = overall_scores.get('pronunciation_score', 0)
        timing_score = overall_scores.get('timing_score', 0)
        pitch_score = overall_scores.get('pitch_score', 0)
        
        # 가중치 적용하여 최종 점수 계산
        final_score = (
            pronunciation_score * SCORING_WEIGHTS['pronunciation'] / 100 +
            timing_score * SCORING_WEIGHTS['timing'] / 100 +
            pitch_score * SCORING_WEIGHTS['pitch'] / 100
        )
        
        # 세부 분석
        textgrid_analysis = comparison_results.get('textgrid_analysis', {})
        pitch_analysis = comparison_results.get('pitch_analysis', {})
        
        result = {
            "overall_score": round(final_score, 1),
            "component_scores": {
                "pronunciation_score": round(pronunciation_score, 1),
                "timing_score": round(timing_score, 1),
                "pitch_score": round(pitch_score, 1)
            },
            "weights_used": SCORING_WEIGHTS,
            "detailed_analysis": {
                "pronunciation_details": {
                    "accuracy": textgrid_analysis.get('pronunciation_accuracy', 0),
                    "matched_phones": textgrid_analysis.get('matched_phones', 0),
                    "total_phones": textgrid_analysis.get('reference_phones', 0)
                },
                "timing_details": {
                    "accuracy": textgrid_analysis.get('timing_accuracy', 0),
                    "reference_duration": textgrid_analysis.get('reference_duration', 0),
                    "user_duration": textgrid_analysis.get('user_duration', 0)
                },
                "pitch_details": {
                    "similarity": pitch_analysis.get('pitch_similarity', 0),
                    "segments_analyzed": len(pitch_analysis.get('segment_details', []))
                }
            }
        }
        
        print(f"📊 최종 점수: {final_score:.1f}/100")
        print(f"   - 발음: {pronunciation_score:.1f} (가중치 {SCORING_WEIGHTS['pronunciation']}%)")
        print(f"   - 타이밍: {timing_score:.1f} (가중치 {SCORING_WEIGHTS['timing']}%)")
        print(f"   - 피치: {pitch_score:.1f} (가중치 {SCORING_WEIGHTS['pitch']}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ 점수 계산 실패: {e}")
        return {
            "overall_score": 0,
            "component_scores": {"pronunciation_score": 0, "timing_score": 0, "pitch_score": 0},
            "error": str(e)
        }

def save_score_result(scores: dict, output_path: str) -> str:
    """
    점수 결과를 JSON 파일로 저장
    
    Args:
        scores: 점수 결과 딕셔너리
        output_path: 저장할 파일 경로
    
    Returns:
        저장된 파일 경로
    """
    from datetime import datetime
    
    results = {
        "scores": scores,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return output_path
