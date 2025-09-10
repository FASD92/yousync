#!/usr/bin/env python3
"""
Mel Spectrogram 유사도 테스트 스크립트
"""

from mel_spectrogram_tester import test_mel_similarity_from_s3

def main():
    print("🎵 Mel Spectrogram 유사도 테스트")
    print("=" * 50)
    
    # 실제 S3 URL을 입력하세요
    reference_url = input("기준 음성 S3 URL을 입력하세요: ").strip()
    user_url = input("유저 음성 S3 URL을 입력하세요: ").strip()
    
    if not reference_url or not user_url:
        print("❌ URL을 모두 입력해주세요.")
        return
    
    try:
        print(f"\n🚀 테스트 시작...")
        print(f"📍 기준 음성: {reference_url}")
        print(f"📍 유저 음성: {user_url}")
        
        results = test_mel_similarity_from_s3(reference_url, user_url)
        
        print("\n" + "=" * 50)
        print("🎯 최종 유사도 결과:")
        print("=" * 50)
        
        similarity_scores = results['similarity_scores']
        
        print(f"📊 코사인 유사도 (평균): {similarity_scores.get('cosine_similarity_mean', 'N/A'):.4f}")
        print(f"📊 코사인 유사도 (표준편차): {similarity_scores.get('cosine_similarity_std', 'N/A'):.4f}")
        print(f"📊 코사인 유사도 (종합): {similarity_scores.get('cosine_similarity_combined', 'N/A'):.4f}")
        print(f"📊 유클리드 유사도: {similarity_scores.get('euclidean_similarity', 'N/A'):.4f}")
        print(f"📊 상관계수: {similarity_scores.get('correlation', 'N/A'):.4f}")
        print(f"🎯 전체 유사도: {similarity_scores.get('overall_similarity', 'N/A'):.4f}")
        
        print(f"\n📁 결과 파일:")
        print(f"  - JSON: mel_test_results/similarity_results.json")
        print(f"  - 시각화: {results['visualization_path']}")
        
        # 유사도 해석
        overall_sim = similarity_scores.get('overall_similarity', 0)
        if overall_sim >= 0.9:
            print(f"\n✅ 매우 높은 유사도! 거의 동일한 음성입니다.")
        elif overall_sim >= 0.7:
            print(f"\n🟢 높은 유사도! 비슷한 음성입니다.")
        elif overall_sim >= 0.5:
            print(f"\n🟡 보통 유사도. 어느 정도 비슷합니다.")
        elif overall_sim >= 0.3:
            print(f"\n🟠 낮은 유사도. 다소 다른 음성입니다.")
        else:
            print(f"\n🔴 매우 낮은 유사도. 많이 다른 음성입니다.")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
