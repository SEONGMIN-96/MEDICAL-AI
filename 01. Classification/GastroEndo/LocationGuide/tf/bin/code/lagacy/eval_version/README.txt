train_u.py
: 모델 학습 코드

eval_u.py
: 모델 평가 코드
Args:
	--exp = best_model 폴더

eval_video.v1.0.1
: 동영상 적용 코드
Args:
	--exp = 메인 분류 best_model 파일 경로
	--video_path = 적용할 비디오 파일 경로

eval_video.v1.0.2
: 동영상 적용 코드
Args:
	--exp_main = 메인 분류 best_model 파일 경로
	--exp_es = 세부 분류[ES-GE] best_model 폴더
	--exp_du = 세부 분류[BB-SD] best_model 폴더
	--video_path = 적용할 비디오 파일 경로

eval_video.v1.0.3
: 동영상 적용 코드
: 멀티 모델 적용
Args:
	--exp_main = 메인 분류 best_model 파일 경로
	--exp_es = 세부 분류[ES-GE] best_model 폴더
	--exp_du = 세부 분류[BB-SD] best_model 폴더
	--video_path = 적용할 비디오 파일 경로

eval_video.v1.0.4
: y_true, y_pred 비교 추가 및 roc_curve추가
Args:
	--exp_main = 메인 분류 best_model 파일 경로
	--exp_es = 세부 분류[ES-GE] best_model 폴더
	--exp_du = 세부 분류[BB-SD] best_model 폴더
	--video_path = 적용할 비디오 파일 경로

eval_video.v1.0.5
: sensitivity, specifity 추가
: eval 구성 요소 저장 추가
Args:
	--exp_main = 메인 분류 best_model 파일 경로
	--exp_es = 세부 분류[ES-GE] best_model 폴더
	--exp_du = 세부 분류[BB-SD] best_model 폴더
	--video_path = 적용할 비디오 파일 경로