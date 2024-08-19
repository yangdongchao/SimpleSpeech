
# A demo recipe for tts based on LibriTTS dataset.
# tts: phone + prompt ---> wave
. ./path.sh


. utils/parse_options.sh
ngpu=4
split_scp=
for n in `seq 1 $ngpu`; do
    split_scp="$split_scp /home/jupyter/data/mls_english_opus/split4/wav.${n}.scp"
done
utils/split_scp.pl /home/jupyter/data/mls_english_opus/wav28.scp $split_scp


utils/run.pl JOB=1:$ngpu ./log/audio_codec_dump.JOB.log \
      python3 asr.py \
        --input-file /home/jupyter/data/mls_english_opus/split4/wav.JOB.scp \
        --output-file /home/jupyter/data/mls_english_opus/text4_out/text.JOB.scp \
        --rank JOB || exit 1;

### stage 1-3: data preparation ###

