
# python scripts/infer_prompt.py --scale 5  --vocoder-ckpt none \
#     -b configs/tts.yaml \
#     --outdir ./results -r "checkpoints/last.ckpt" \
#     --prompt "For the past ten years, Conseil had gone with me wherever science beckoned." \
#     --W  225 --H 32 \
#     --prompt_audio /home/jupyter/tmp_code/simplespeech_demo/ns3/ns3_p6.wav

python scripts/infer_prompt2.py --scale 5  --vocoder-ckpt  none \
    -b configs/tts.yaml \
    --outdir /results -r "checkpoints/last.ckpt" \
    --test_file test.scp

