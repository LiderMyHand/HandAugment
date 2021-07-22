# stage1
echo 'stage1 start'
python test_hands19task1.py \
  --output_result_path ${result_path1} \
  -v \
  --input_test_img_folder cache/hands19task1/test_images_augment \
  --model_path weights/stage1.pth \
  --gpu_id 0 \
  --batch_size 128
cd ${dst_dir1}
zip -q result.txt.zip result.txt
echo 'stage1 done'
