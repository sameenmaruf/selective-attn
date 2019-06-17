For pre-training the sentence-level transformer: (Here the data files should be in format source ||| target and --model-path provides path to folder where model files will be save)

./build_gpu/transformer-train --dynet_mem 15500 --minibatch-size 1000 --treport 8000 --dreport 40000 -t $trainfname -d $devfname --model-path $modelfname \
--sgd-trainer 4 --lr-eta 0.0001 -e 35 --patience 10 --use-label-smoothing --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 \
--decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 8

For computing encoder-based (monolingual) representations: (Here input_doc is in format docID ||| source, input_type denotes the portion of data i.e. 0 for train, 1 for dev)

./build_gpu/transformer-computerep --dynet-mem 15500 --model-path $modelfname --input_doc $fname --input_type 0 --rep_type 1

For computing decoder-based (bilingual) representations: (Here input_doc is in format docID ||| source, --input_type denotes the portion of data i.e. 0 for train, 1 for dev)

./build_gpu/transformer-computerep --dynet-mem 15500 --model-path $modelfname --input_doc $fname --input_type 0 --rep_type 2

For training the document-level model with hierarchical attention: (Here the data files should be in format docID ||| source ||| target, --model-file is to give the model a name of your choice, 
--context-type is 1 for monolingual and 2 for bilingual context, --use-sparse-soft is 1 for sparse at sentence and soft at word-level)

./build_gpu/transformer-context --dynet_mem 15500 --dynet-devices GPU:0,CPU --minibatch-size 1000 --dtreport 65 --ddreport 325 --train_doc $trainfname --devel_doc $devfname \
--model-path $modelfname --model-file $modelname --context-type 1 --doc-attention-type 3 --use-sparse-soft 1 --use-new-dropout \
--encoder-emb-dropout-p 0.2 --encoder-sublayer-dropout-p 0.2 --decoder-emb-dropout-p 0.2 --decoder-sublayer-dropout-p 0.2 --attention-dropout-p 0.2 --ff-dropout-p 0.2 \
--sgd-trainer 4 --lr-eta 0.0001 -e 35 --patience 10 

Decoding the sentence-level Transformer: (Here the test file is in format docID ||| source, only greedy decoding has been implemented)

./build_gpu/transformer-decode --dynet-mem 15500 --model-path $modelfname --beam 1 -T $testfname 

Decoding the document-level model with hierarchical attention: (For decoding the representations are computed on the fly)

./build_gpu/transformer-context-decode --dynet-mem 15500 --dynet-devices GPU:0,CPU --model-path $modelfname --model-file $modelname --beam 1 -T $testfname --context-type 1

(Note: need to set --dynet-devices only when using sparsemax)