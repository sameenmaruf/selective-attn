/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
* Extended for Context-based NMT by Sameen Maruf
*/

#include "transformer-context.h"

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <sys/stat.h>

// Boost
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

// MTEval
#include <mteval/utils.h>
#include <mteval/Evaluator.h>
#include <mteval/EvaluatorFactory.h>
#include <mteval/Statistics.h>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace MTEval;
using namespace boost::program_options;

// hyper-paramaters for training
unsigned MINIBATCH_SIZE = 1;

bool DEBUGGING_FLAG = false;

bool PRINT_GRAPHVIZ = false;

unsigned DTREPORT = 500;
unsigned DDREPORT = 2000;

bool SAMPLING_TRAINING = false;

bool RESET_IF_STUCK = false;
bool SWITCH_TO_ADAM = false;
bool USE_SMALLER_MINIBATCH = false;
unsigned NUM_RESETS = 1;

bool VERBOSE = false;

// ---
bool load_data(const variables_map& vm
	, DocCorpus& train_doccor, DocCorpus& devel_doccor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm);
// ---
void load_wordrep(vector<vector<vector<vector<dynet::real>>>> &sent_rep, const string &rep_file);
void load_tgtrep(vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &sent_rep, const string &rep_file);
// ---
void save_config(const std::string& config_out_file, const std::string& params_out_file
, const TransformerConfig& tfc);
// ---

//---
std::string get_sentence(const WordIdSentence& source, Dict& td);
//---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model);
// ---

// ---
void run_monotrain(transformer::TransformerContextModel &tf, const DocCorpus &train_doccor, const DocCorpus &devel_doccor,
               vector<vector<vector<vector<dynet::real>>>> &tsrcwordrep_cor, vector<vector<vector<vector<dynet::real>>>> &dsrcwordrep_cor,
               dynet::Trainer*& p_sgd,
               const std::string& params_out_file,
               unsigned max_epochs, unsigned patience,
               unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
               unsigned average_checkpoints,
               unsigned dev_eval_mea, unsigned dev_eval_infer_algo, unsigned update_steps);//supports batching
void run_bilingualtrain(transformer::TransformerContextModel &tf, const DocCorpus &train_doccor, const DocCorpus &devel_doccor,
                        vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &ttgtwordrep_cor, vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &dtgtwordrep_cor,
                        dynet::Trainer*& p_sgd,
                        const std::string& params_out_file,
                        unsigned max_epochs, unsigned patience,
                        unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
                        unsigned average_checkpoints,
                        unsigned dev_eval_mea, unsigned dev_eval_infer_algo, unsigned update_steps);//supports batching
// ---

// ---
void get_dev_stats(const DocCorpus &devel_doccor
        , const transformer::TransformerConfig& tfc
        , transformer::ModelStats& dstats);
//void eval_on_dev(transformer::TransformerModel &tf,
//	const WordIdCorpus &devel_cor,
//	transformer::ModelStats& dstats,
//	unsigned dev_eval_mea, unsigned dev_eval_infer_algo);
void eval_on_dev(transformer::TransformerContextModel &tf,
                 const std::vector<std::vector<WordIdSentences>> &dev_src_docminibatch, const std::vector<std::vector<WordIdSentences>> &dev_tgt_docminibatch,
                 vector<vector<vector<vector<dynet::real>>>> &dsrcwordrep_cor, std::vector<std::vector<std::vector<unsigned int>>> dev_sids_docminibatch,
                 transformer::ModelStats& dstats,
                 unsigned dev_eval_mea, unsigned dev_eval_infer_algo); // batched version of eval_on_dev
void eval_on_dev(transformer::TransformerContextModel &tf,
                 const std::vector<std::vector<WordIdSentences>> &dev_src_docminibatch, const std::vector<std::vector<WordIdSentences>> &dev_trg_docminibatch,
                 vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &dtgtwordrep_cor, std::vector<std::vector<std::vector<unsigned int>>> dev_sids_docminibatch,
                 transformer::ModelStats& dstats,
                 unsigned dev_eval_mea, unsigned dev_eval_infer_algo); // batched version of eval_on_dev
// ---

//************************************************************************************************************************************************************
int main(int argc, char** argv) {
	cerr << "*** DyNet initialization ***" << endl;
	auto dyparams = dynet::extract_dynet_params(argc, argv);
	dynet::initialize(dyparams);	

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<std::string>(), "config file specifying additional command line options")
		//-----------------------------------------
		("train_doc", value<std::string>(), "file containing training sentences, with each line consisting of docID ||| source ||| target.")
		("devel_doc", value<std::string>(), "file containing development sentences.")
		("src-vocab", value<std::string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("tgt-vocab", value<std::string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")
		//-----------------------------------------
		("minibatch-size,b", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); single batch by default")
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("sgd-trainer", value<unsigned>()->default_value(4), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp; 6: cyclical SGD)")
		("sparse-updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		("grad-clip-threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
		//-----------------------------------------
		("model-path", value<std::string>()->default_value("."), "all files related to the model will be saved in this folder")
		("model-file", value<std::string>(), "name of model file without extension")
		("model-file-init", value<std::string>(), "name of model file without extension for initialisation in case of incremental training")
		//-----------------------------------------
        // new layers/dimensions should not be defined here, only the hyperparameters
		//-----------------------------------------
		("use-new-dropout", "use a different dropout rate than base model")
		("encoder-emb-dropout-p", value<float>()->default_value(0.1f), "use dropout for encoder embeddings; 0.1 by default")
		("encoder-sublayer-dropout-p", value<float>()->default_value(0.1f), "use dropout for sub-layer's output in encoder; 0.1 by default")
		("decoder-emb-dropout-p", value<float>()->default_value(0.1f), "use dropout for decoding embeddings; 0.1 by default")
		("decoder-sublayer-dropout-p", value<float>()->default_value(0.1f), "use dropout for sub-layer's output in decoder; 0.1 by default")
		("attention-dropout-p", value<float>()->default_value(0.1f), "use dropout for attention; 0.1 by default")
		("ff-dropout-p", value<float>()->default_value(0.1f), "use dropout for feed-forward layer; 0.1 by default")
		//-----------------------------------------
		//("use-label-smoothing", "use label smoothing for cross entropy; no by default")
		//("label-smoothing-weight", value<float>()->default_value(0.1f), "impose label smoothing weight in objective function; 0.1 by default")
		//-----------------------------------------
		//("ff-activation-type", value<unsigned>()->default_value(1), "impose feed-forward activation type (1: RELU, 2: SWISH, 3: SWISH with learnable beta); 1 by default")
		//-----------------------------------------
		//("position-encoding", value<unsigned>()->default_value(2), "impose positional encoding (0: none; 1: learned positional embedding; 2: sinusoid encoding); 2 by default")
		//("position-encoding-flag", value<unsigned>()->default_value(0), "which both (0) / encoder only (1) / decoder only (2) will be applied positional encoding; both (0) by default")
		//("max-pos-seq-len", value<unsigned>()->default_value(300), "specify the maximum word-based sentence length (either source or target) for learned positional encoding; 300 by default")
		//-----------------------------------------
		//("use-hybrid-model", "use hybrid model in which RNN encodings of source and target are used in place of word embeddings and positional encodings (a hybrid architecture between AM and Transformer?) partially adopted from GNMT style; no by default")
		//-----------------------------------------
		//("attention-type", value<unsigned>()->default_value(1), "impose attention type (1: Luong attention type; 2: Bahdanau attention type); 1 by default")
		//-----------------------------------------
		("epochs,e", value<unsigned>()->default_value(20), "maximum number of training epochs")
		("patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved for early stopping; default none")
		//-----------------------------------------
		("lr-eta", value<float>()->default_value(0.0001f), "SGD learning rate value (e.g., 0.1 for simple SGD trainer or smaller 0.0001 for ADAM trainer)")
		("lr-eta-decay", value<float>()->default_value(2.0f), "SGD learning rate decay value")
		//-----------------------------------------
		// learning rate scheduler
        ("lr-epochs", value<unsigned>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)") // learning rate scheduler 1
		("lr-patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved, e.g., for starting learning rate annealing (e.g., halving)") // learning rate scheduler 2
		//-----------------------------------------
		//these options have not been tested out
		("reset-if-stuck", "a strategy if the model gets stuck then reset everything and resume training; default not")
		("switch-to-adam", "switch to Adam trainer if getting stuck; default not")
		("use-smaller-minibatch", "use smaller mini-batch size if getting stuck; default not")
		("num-resets", value<unsigned>()->default_value(1), "no. of times the training process will be reset; default 1") 
		//-----------------------------------------
		("sampling", "sample translation during training; default not")
		//-----------------------------------------
		("dev-eval-measure", value<unsigned>()->default_value(0), "specify measure for evaluating dev data during training (0: perplexity); default 0 (perplexity) has only been used") // note that MT scores here are approximate (e.g., evaluating with <unk> markers, and tokenized text or with subword segmentation if using BPE), not necessarily equivalent to real BLEU/NIST/WER/RIBES scores.
		("dev-eval-infer-algo", value<unsigned>()->default_value(1), "specify the algorithm for inference on dev (0: sampling; 1: greedy; N>=2: beam search with N size of beam); default 0 (sampling) never used") // using sampling/greedy will be faster.
		//-----------------------------------------
		("average-checkpoints", value<unsigned>()->default_value(1), "specify number of checkpoints for model averaging; default single best model, never used") // average checkpointing
		//-----------------------------------------
        ("online-docmt", "if this is set do Online DocumentMT else just assume that complete document is given")
        ("context-type", value<unsigned>()->default_value(1), "decide which type of context to use (1: monolingual from encoder, 2: bilingual from decoder)")
        ("doc-attention-type", value<unsigned>()->default_value(1), "impose attention type (1: sentence-level attention; 2: word-level attention; 3: hierarchical attention); 1 by default at the moment")
        ("use-sparse-soft", value<unsigned>()->default_value(1), "impose sparse/softmax for attention (1: sparse at sentence-level, soft at word-level, 2: sparse at both levels); 1 by default at the moment")
        ("update-steps", value<unsigned>()->default_value(1), "number of documents after which to update the model; may be needed for short documents, dtreport should be divisible by this")
        //-----------------------------------------
        //("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		//("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("dtreport", value<unsigned>()->default_value(50), "no. of training documents for reporting current model status on training data")
		("ddreport", value<unsigned>()->default_value(200), "no. of training documents for reporting current model status on development data (dreport = N * treport)")
		//-----------------------------------------
		("print-graphviz", "print graphviz-style computation graph; default not")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		//-----------------------------------------
		("debug", "enable/disable simpler debugging by immediate computing mode or checking validity (refers to http://dynet.readthedocs.io/en/latest/debugging.html)")// for CPU only
		("dynet-profiling", value<int>()->default_value(0), "enable/disable auto profiling (https://github.com/clab/dynet/pull/1088/commits/bc34db98fa5e2e694f54f0e6b1d720d517c7530e)")// for debugging only			
	;
	
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<std::string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);

	// print command line
	cerr << endl << "PID=" << ::getpid() << endl;
	cerr << "Command: ";
	for (int i = 0; i < argc; i++){ 
		cerr << argv[i] << " "; 
	} 
	cerr << endl;
	
	// print help
	if (vm.count("help") 
		|| !(vm.count("train_doc") && vm.count("devel_doc")))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// hyper-parameters for training
	DEBUGGING_FLAG = vm.count("debug");
	VERBOSE = vm.count("verbose");
	DTREPORT = vm["dtreport"].as<unsigned>();
	DDREPORT = vm["ddreport"].as<unsigned>();
	SAMPLING_TRAINING = vm.count("sampling");
	PRINT_GRAPHVIZ = vm.count("print-graphviz");
	RESET_IF_STUCK = vm.count("reset-if-stuck");
	SWITCH_TO_ADAM = vm.count("switch-to-adam");
	USE_SMALLER_MINIBATCH = vm.count("use-smaller-minibatch");
	NUM_RESETS = vm["num-resets"].as<unsigned>();
	MINIBATCH_SIZE = vm["minibatch-size"].as<unsigned>();

    if (DTREPORT % vm["update-steps"].as<unsigned>() != 0) assert("dtreport must be divisible by update-steps else may run into issues.");// to ensure not reporting on development data before updating params

    // get and check model path
	std::string model_path = vm["model-path"].as<std::string>();
	struct stat sb;
	if (stat(model_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		cerr << endl << "All model files will be saved to: " << model_path << "." << endl;
	else
		TRANSFORMER_RUNTIME_ASSERT("The model-path does not exist!");

	// model recipe
	dynet::Dict sd, td;// vocabularies
	SentinelMarkers sm;// sentinel markers
	DocCorpus train_doccor, devel_doccor;// document-level corpus
    transformer::TransformerConfig tfc;// Transformer's configuration (either loaded from file or newly-created)

	if (!vm.count("model-file-init")) {
		std::string config_file = model_path + "/base-model.config";// configuration file path
		if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {// check existence
			// (incremental training)
			// to load the training profiles from previous training run
			cerr << "Found existing (trained) model from " << model_path << "!" << endl;

			// load vocabulary from files
			std::string vocab_file = model_path + "/" + "src-tgt.joint-vocab";
			if (stat(vocab_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
				load_vocab(model_path + "/" + "src-tgt.joint-vocab", sd);
				td = sd;
			} else {
				std::string src_vocab_file = model_path + "/" + "src.vocab";
				std::string tgt_vocab_file = model_path + "/" + "tgt.vocab";
				load_vocabs(src_vocab_file, tgt_vocab_file, sd, td);
			}

			// initalise sentinel markers
			sm._kSRC_SOS = sd.convert("<s>");
			sm._kSRC_EOS = sd.convert("</s>");
			sm._kSRC_UNK = sd.convert("<unk>");
			sm._kTGT_SOS = td.convert("<s>");
			sm._kTGT_EOS = td.convert("</s>");
			sm._kTGT_UNK = td.convert("<unk>");

			// load data files
			if (!load_data(vm, train_doccor, devel_doccor, sd, td, sm))
				TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

			// read model configuration
			//cerr << "Reading the config file of base Transformer" << endl;
			ifstream inpf_cfg(config_file);
			assert(inpf_cfg);

			std::string line;
			getline(inpf_cfg, line);
			std::stringstream ss(line);
			tfc._src_vocab_size = sd.size();
			tfc._tgt_vocab_size = td.size();
			tfc._sm = sm;
			ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
			   >> tfc._encoder_emb_dropout_rate >> tfc._encoder_sublayer_dropout_rate >> tfc._decoder_emb_dropout_rate
			   >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate
			   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
			   >> tfc._position_encoding >> tfc._position_encoding_flag >> tfc._max_length
			   >> tfc._attention_type
			   >> tfc._ffl_activation_type
			   >> tfc._shared_embeddings
			   >> tfc._use_hybrid_model;

			if (vm.count("use-new-dropout")) {
				tfc._encoder_emb_dropout_rate = vm["encoder-emb-dropout-p"].as<float>();
				tfc._encoder_sublayer_dropout_rate = vm["encoder-sublayer-dropout-p"].as<float>();
				tfc._decoder_emb_dropout_rate = vm["decoder-emb-dropout-p"].as<float>();
				tfc._decoder_sublayer_dropout_rate = vm["decoder-sublayer-dropout-p"].as<float>();
				tfc._attention_dropout_rate = vm["attention-dropout-p"].as<float>();
				tfc._ff_dropout_rate = vm["ff-dropout-p"].as<float>();
			}

			tfc._online_docmt = vm.count("online-docmt");
			tfc._doc_attention_type = vm["doc-attention-type"].as<unsigned>();
			tfc._context_type = vm["context-type"].as<unsigned>();
			tfc._use_sparse_soft = vm["use-sparse-soft"].as<unsigned>();

			// save new configuration file (for decoding/inference)
			std::string config_out_file = model_path + "/" + vm["model-file"].as<std::string>() + ".config";
			std::string params_out_file = model_path + "/" + vm["model-file"].as<std::string>() + ".params";
			save_config(config_out_file, params_out_file, tfc);
		} else// not exist!
			TRANSFORMER_RUNTIME_ASSERT("No pre-trained model found in path...");
	}
	else{
		std::string config_file = model_path + "/" + vm["model-file-init"].as<std::string>() + ".config";// configuration file path
		if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {// check existence
			// (incremental training)
			// to load the training profiles from previous training run
			cerr << "Found existing (trained) model from " << model_path << "!" << endl;

			// load vocabulary from files
			std::string vocab_file = model_path + "/" + "src-tgt.joint-vocab";
			if (stat(vocab_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
				load_vocab(model_path + "/" + "src-tgt.joint-vocab", sd);
				td = sd;
			} else {
				std::string src_vocab_file = model_path + "/" + "src.vocab";
				std::string tgt_vocab_file = model_path + "/" + "tgt.vocab";
				load_vocabs(src_vocab_file, tgt_vocab_file, sd, td);
			}

			// initalise sentinel markers
			sm._kSRC_SOS = sd.convert("<s>");
			sm._kSRC_EOS = sd.convert("</s>");
			sm._kSRC_UNK = sd.convert("<unk>");
			sm._kTGT_SOS = td.convert("<s>");
			sm._kTGT_EOS = td.convert("</s>");
			sm._kTGT_UNK = td.convert("<unk>");

			// load data files
			if (!load_data(vm, train_doccor, devel_doccor, sd, td, sm))
				TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

			// read model configuration
			//cerr << "Reading the config file of base Transformer" << endl;
			ifstream inpf_cfg(config_file);
			assert(inpf_cfg);

			std::string line;
			getline(inpf_cfg, line);
			std::stringstream ss(line);
			tfc._src_vocab_size = sd.size();
			tfc._tgt_vocab_size = td.size();
			tfc._sm = sm;
			ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
			   >> tfc._encoder_emb_dropout_rate >> tfc._encoder_sublayer_dropout_rate >> tfc._decoder_emb_dropout_rate >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate
			   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
			   >> tfc._position_encoding >> tfc._position_encoding_flag >> tfc._max_length
			   >> tfc._attention_type
			   >> tfc._ffl_activation_type
			   >> tfc._shared_embeddings
			   >> tfc._use_hybrid_model
			   >> tfc._online_docmt
			   >> tfc._doc_attention_type
			   >> tfc._context_type
			   >> tfc._use_sparse_soft;

			// save new configuration file (for decoding/inference)
			std::string config_out_file = model_path + "/" + vm["model-file"].as<std::string>() + ".config";
			std::string params_out_file = model_path + "/" + vm["model-file"].as<std::string>() + ".params";
			save_config(config_out_file, params_out_file, tfc);
		} else// not exist!
			TRANSFORMER_RUNTIME_ASSERT("No pre-trained model found in path...");
	}

    // learning rate scheduler
	unsigned lr_epochs = vm["lr-epochs"].as<unsigned>(), lr_patience = vm["lr-patience"].as<unsigned>();
	if (lr_epochs > 0 && lr_patience > 0)
		cerr << "[WARNING] - Conflict on learning rate scheduler; use either lr-epochs or lr-patience!" << endl;

	// initialise transformer object
    transformer::TransformerContextModel tf(tfc, sd, td);
	if (!vm.count("model-file-init")) {
		std::string model_file = model_path + "/base-model.params";
		if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
			cerr << endl << "Loading pre-trained model from file: " << model_file << "..." << endl;
			tf.initialise_baseparams_from_file(model_file);// load pre-trained model (for document-level training)
		}
	}
	else{
		std::string model_file = model_path + "/" + vm["model-file-init"].as<std::string>() + ".params";
		if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
			cerr << endl << "Loading pre-trained model from file: " << model_file << "..." << endl;
			tf.initialise_params_from_file(model_file);// load pre-trained model (for document-level incremental training)
		}
	}
	cerr << "Count of model parameters: " << tf.get_model_parameters().parameter_count() << endl;

    std::string rep_path;// representation file path
    vector<vector<vector<vector<dynet::real>>>> tsrcwordrep_corpus, dsrcwordrep_corpus;
    vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> ttgtwordrep_corpus, dtgtwordrep_corpus;
    if (tfc._context_type == CONTEXT_TYPE::MONOLINGUAL) {
        rep_path = model_path + "/src-rep";

        if (stat(rep_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
            cerr << endl << "The source representations will be loaded from: " << rep_path << "." << endl;
        else
            TRANSFORMER_RUNTIME_ASSERT("The source representations path does not exist!");

        // get the source representations from the encoder
        std::string train_file = rep_path + "/trainrep.bin";
        if (stat(train_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence
            cerr << "*Loading source representations of training set from " << train_file << endl;
            load_wordrep(tsrcwordrep_corpus, train_file);
        }
        else
            TRANSFORMER_RUNTIME_ASSERT("No source representations of training set found!");

        std::string dev_file = rep_path + "/devrep.bin";
        if (stat(dev_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence
            cerr << "*Loading source representations of development set from " << dev_file << endl;
            load_wordrep(dsrcwordrep_corpus, dev_file);
        }
        else
            TRANSFORMER_RUNTIME_ASSERT("No source representations of development set found!");
    }
    else{
        rep_path = model_path + "/tgt-rep";

        if (stat(rep_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
            cerr << endl << "The bilingual representations will be loaded from: " << rep_path << "." << endl;
        else
            TRANSFORMER_RUNTIME_ASSERT("The bilingual representations path does not exist!");

        // get the bilingual representations from the decoder
        std::string train_file = rep_path + "/trainrep.bin";
        if (stat(train_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence
            cerr << "*Loading bilingual representations of training set from " << train_file << endl;
            load_tgtrep(ttgtwordrep_corpus, train_file);
        }
        else
            TRANSFORMER_RUNTIME_ASSERT("No bilingual representations of training set found!");

        std::string dev_file = rep_path + "/devrep.bin";
        if (stat(dev_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence
            cerr << "*Loading bilingual representations of development set from " << dev_file << endl;
            load_tgtrep(dtgtwordrep_corpus, dev_file);
        }
        else
            TRANSFORMER_RUNTIME_ASSERT("No bilingual representations of development set found!");
    }

    // create SGD trainer
	Trainer* p_sgd_trainer = create_sgd_trainer(vm, tf.get_model_parameters());

	if (vm["dev-eval-measure"].as<unsigned>() > 4) TRANSFORMER_RUNTIME_ASSERT("Unknown dev-eval-measure type (0: perplexity; 1: BLEU; 2: NIST; 3: WER; 4: RIBES)!");

	// model params file
	std::string params_out_file = model_path + "/" + vm["model-file"].as<std::string>() + ".params";

	// train transformer model
    if (tfc._context_type == CONTEXT_TYPE::MONOLINGUAL) {
        run_monotrain(tf, train_doccor, devel_doccor, tsrcwordrep_corpus, dsrcwordrep_corpus, p_sgd_trainer,
                  params_out_file, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>()/*early stopping*/
                , lr_epochs, vm["lr-eta-decay"].as<float>(), lr_patience/*learning rate scheduler*/
                , vm["average-checkpoints"].as<unsigned>(), vm["dev-eval-measure"].as<unsigned>(),
                  vm["dev-eval-infer-algo"].as<unsigned>(), vm["update-steps"].as<unsigned>());
    }
    else{
        run_bilingualtrain(tf, train_doccor, devel_doccor, ttgtwordrep_corpus, dtgtwordrep_corpus, p_sgd_trainer,
                      params_out_file, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>()/*early stopping*/
                      , lr_epochs, vm["lr-eta-decay"].as<float>(), lr_patience/*learning rate scheduler*/
                      , vm["average-checkpoints"].as<unsigned>(), vm["dev-eval-measure"].as<unsigned>(),
                      vm["dev-eval-infer-algo"].as<unsigned>(), vm["update-steps"].as<unsigned>());
    }

	// clean up
	cerr << "Cleaning up..." << endl;
	delete p_sgd_trainer;
	// transformer object will be automatically cleaned, no action required!

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
bool load_data(const variables_map& vm
	, DocCorpus& train_doccor, DocCorpus& devel_doccor
	, dynet::Dict& sd, dynet::Dict& td
	, SentinelMarkers& sm)
{
	SentCorpus train_sentcor, devel_sentcor;
	std::string train_path = vm["train_doc"].as<std::string>();// single training data file
	cerr << endl << "Reading training data from " << train_path << "...\n";
	bool use_joint_vocab = vm.count("joint-vocab") | vm.count("shared-embeddings"); 
	if (use_joint_vocab){
		train_sentcor = read_doccorpus(train_path, &sd, &sd);
        train_doccor = read_doccorpus(train_sentcor);
		td = sd;
	}
	else {
        train_sentcor = read_doccorpus(train_path, &sd, &td);
        train_doccor = read_doccorpus(train_sentcor);
    }

    if ("" == vm["src-vocab"].as<std::string>()
		&& "" == vm["tgt-vocab"].as<std::string>()) // if not using external vocabularies
	{
		sd.freeze(); // no new word types allowed
		td.freeze(); // no new word types allowed
	}

	if (DDREPORT >= train_doccor.size())
		cerr << "WARNING: --dreport <num> (" << DDREPORT << ")" << " is too large, <= training data size (" << train_doccor.size() << ")" << endl;

	// set up <unk> ids
	sd.set_unk("<unk>");
	sm._kSRC_UNK = sd.get_unk_id();
	td.set_unk("<unk>");
	sm._kTGT_UNK = td.get_unk_id();

	if (vm.count("devel_doc")) {
		cerr << "Reading dev data from " << vm["devel_doc"].as<std::string>() << "...\n";
		devel_sentcor = read_doccorpus(vm["devel_doc"].as<std::string>(), &sd, &td);
        devel_doccor = read_doccorpus(devel_sentcor);
    }
    /*
	if (swap) {
		cerr << "Swapping role of source and target\n";
		if (!use_joint_vocab){
			std::swap(sd, td);
			std::swap(sm._kSRC_SOS, sm._kTGT_SOS);
			std::swap(sm._kSRC_EOS, sm._kTGT_EOS);
			std::swap(sm._kSRC_UNK, sm._kTGT_UNK);
		}

		for (auto &sent: train_sentcor)
			std::swap(get<0>(sent), get<1>(sent));
        train_doccor.clear();
        train_doccor = read_doccorpus(train_sentcor);

		for (auto &sent: devel_sentcor)
			std::swap(get<0>(sent), get<1>(sent));
        devel_doccor.clear();
        devel_doccor = read_doccorpus(devel_sentcor);
    }
     */

	return true;
}
// ---
void load_wordrep(vector<vector<vector<vector<dynet::real>>>> &sent_rep, const string &rep_file)
{
    ifstream in(rep_file);
    boost::archive::binary_iarchive ia(in);

    ia >> sent_rep;
    in.close();
}

void load_tgtrep(vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &sent_rep, const string &rep_file)
{
    ifstream in(rep_file);
    boost::archive::binary_iarchive ia(in);

    ia >> sent_rep;
    in.close();
}
// ---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model){
	// setup SGD trainer
	Trainer* sgd = nullptr;
	unsigned sgd_type = vm["sgd-trainer"].as<unsigned>();
	if (sgd_type == 1)
		sgd = new MomentumSGDTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 5)
		sgd = new RMSPropTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, vm["lr-eta"].as<float>());
	else
	   	TRANSFORMER_RUNTIME_ASSERT("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");
	sgd->clip_threshold = vm["grad-clip-threshold"].as<float>();// * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching, correct?
	sgd->sparse_updates_enabled = vm["sparse-updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	return sgd;
}
// ---

// ---
void get_dev_stats(const DocCorpus &devel_doccor
	, const transformer::TransformerConfig& tfc
	, transformer::ModelStats& dstats)
{
    WordIdSentence ssent, tsent;

    for (unsigned i = 0; i < devel_doccor.size(); ++i) {
        const unsigned ddlen = devel_doccor[i].size();
        for (unsigned dl = 0; dl < ddlen; ++dl){
            ssent = get<0>(devel_doccor[i].at(dl));
            tsent = get<1>(devel_doccor[i].at(dl));

            dstats._words_src += ssent.size();
            dstats._words_tgt += tsent.size() - 1; // shifted right
            for (auto& word : ssent) if (word == tfc._sm._kSRC_UNK) dstats._words_src_unk++;
            for (auto& word : tsent) if (word == tfc._sm._kTGT_UNK) dstats._words_tgt_unk++;
        }
    }
}

//FIXME for document-level
/*
void eval_on_dev(transformer::TransformerModel &tf, 
	const WordIdCorpus &devel_cor, 
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo)
{
	if (dev_eval_mea == 0) // perplexity
	{
		double losses = 0.f;
		for (unsigned i = 0; i < devel_cor.size(); ++i) {
			WordIdSentence ssent, tsent;
			tie(ssent, tsent) = devel_cor[i];  

			dynet::ComputationGraph cg;
			auto i_xent = tf.build_graph(cg, WordIdSentences(1, ssent), WordIdSentences(1, tsent), nullptr, true);
			losses += as_scalar(cg.forward(i_xent));
		}

		dstats._scores[1] = losses;
	}
	else{
		// create evaluators
		std::string spec;
		if (dev_eval_mea == 1) spec = "BLEU";
		else if (dev_eval_mea == 2) spec = "NIST";
		else if (dev_eval_mea == 3) spec = "WER";
		else if (dev_eval_mea == 4) spec = "RIBES";
		std::shared_ptr<MTEval::Evaluator> evaluator(MTEval::EvaluatorFactory::create(spec));
		std::vector<MTEval::Sample> v_samples;
		for (unsigned i = 0; i < devel_cor.size(); ++i) {
			WordIdSentence ssent, tsent;
			tie(ssent, tsent) = devel_cor[i];  

			// inference
			dynet::ComputationGraph cg;
			WordIdSentence thyp;// raw translation (w/o scores)
			if (dev_eval_infer_algo == 0)// random sampling
				tf.sample(cg, ssent, thyp);// fastest with bad translations
			else if (dev_eval_infer_algo == 1)// greedy decoding
				tf.greedy_decode(cg, ssent, thyp);// faster with relatively good translations
			else// beam search decoding
			{
				WordIdSentences thyps;	
				tf.beam_decode(cg, ssent, thyps, dev_eval_infer_algo);// slow with better translations
				thyp = thyps[0];
			}
					
			// collect statistics for mteval
			v_samples.push_back(MTEval::Sample({thyp, {tsent}}));// multiple references are supported as well!
      			evaluator->prepare(v_samples[v_samples.size() - 1]);
		}
		
		// analyze the evaluation score
		MTEval::Statistics eval_stats;
    		for (unsigned i = 0; i < v_samples.size(); ++i) {			
      			eval_stats += evaluator->map(v_samples[i]);
    		}

		dstats._scores[1] = evaluator->integrate(eval_stats);
	}

}*/

void eval_on_dev(transformer::TransformerContextModel &tf,
	const std::vector<std::vector<WordIdSentences>> &dev_src_docminibatch, const std::vector<std::vector<WordIdSentences>> &dev_trg_docminibatch,
    vector<vector<vector<vector<dynet::real>>>> &dsrcwordrep_cor, std::vector<std::vector<std::vector<unsigned int>>> dev_sids_docminibatch,
	transformer::ModelStats& dstats, 
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo) // batched version of eval_on_dev
{
    vector<WordIdSentences> dev_src_doc, dev_trg_doc;
    vector<vector<unsigned int>> dev_doc_sids;

    if (dev_eval_mea == 0) // perplexity
	{
		double losses = 0.f;

        for (unsigned d = 0; d < dev_src_docminibatch.size(); ++d) {
            dev_src_doc = dev_src_docminibatch[d];
            dev_trg_doc = dev_trg_docminibatch[d];
            dev_doc_sids = dev_sids_docminibatch[d];
            vector<vector<vector<dynet::real>>> dsrcwordrep_doc = dsrcwordrep_cor[d];

            for (unsigned i = 0; i < dev_src_doc.size(); ++i) {
                ComputationGraph cg;

                auto i_xent = tf.build_graph(cg, dev_src_doc[i], dev_trg_doc[i], dsrcwordrep_doc, dev_doc_sids[i], nullptr, true);
                losses += as_scalar(cg.forward(i_xent));
            }
        }
        dstats._scores[1] = losses;
	}
}
// ---

// ---
void run_monotrain(transformer::TransformerContextModel &tf, const DocCorpus &train_doccor, const DocCorpus &devel_doccor,
    vector<vector<vector<vector<dynet::real>>>> &tsrcwordrep_cor, vector<vector<vector<vector<dynet::real>>>> &dsrcwordrep_cor,
    dynet::Trainer*& p_sgd,
	const std::string& params_out_file,
	unsigned max_epochs, unsigned patience,
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints,
	unsigned dev_eval_mea, unsigned dev_eval_infer_algo, unsigned update_steps)
{
    unsigned did = 0, id = 0, last_print = 0, lines = 0;
    MyTimer timer_epoch("completed in"), timer_iteration("completed in");
	unsigned epoch = 0, cpt = 0/*count of patience*/;

	// get current configuration
	const transformer::TransformerConfig& tfc = tf.get_config();

	// create minibatches
    std::vector<std::vector<WordIdSentences>> train_src_docminibatch, dev_src_docminibatch;
    std::vector<std::vector<WordIdSentences>> train_trg_docminibatch, dev_trg_docminibatch;
    std::vector<std::vector<std::vector<unsigned int>>> train_sids_docminibatch, dev_sids_docminibatch;
    size_t minibatch_size = MINIBATCH_SIZE;
    cerr << endl << "Creating minibatches for training data (using minibatch_size=" << minibatch_size << ")..." << endl;
    create_docminibatches(train_doccor, minibatch_size, train_src_docminibatch, train_trg_docminibatch, train_sids_docminibatch);// on train
    cerr << "Creating minibatches for development data (using minibatch_size=" << minibatch_size*2/3<< ")..." << endl;
    create_docminibatches(devel_doccor, minibatch_size*2/3, dev_src_docminibatch, dev_trg_docminibatch, dev_sids_docminibatch);// on dev
    // create a document list for the train data
    vector<unsigned> order(train_doccor.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    // model stats on dev
	transformer::ModelStats dstats(dev_eval_mea);
	get_dev_stats(devel_doccor, tfc, dstats);

    // ----
    // pre-compute model score on dev before training
    tf.set_dropout(false);// disable dropout
    //eval_on_dev(tf, devel_doccor, dstats, dev_eval_mea, dev_eval_infer_algo);// non-batched version FIXME for document-level
    eval_on_dev(tf, dev_src_docminibatch, dev_trg_docminibatch, dsrcwordrep_cor, dev_sids_docminibatch, dstats,
                dev_eval_mea, dev_eval_infer_algo);// batched version (2-3 times faster) FIXME for document-level
    float elapsed = timer_iteration.elapsed();
    timer_iteration.reset();
    
    // update best score so far (in case of incremental training)
    dstats.update_best_score(cpt);

    // verbose
    cerr << endl << "--------------------------------------------------------------------------------------------------------" << endl;
    cerr << "***Initial score on DEV: docs=" << devel_doccor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';
    cerr << "[completed in " << elapsed << " ms]" << endl;
    cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    // ----

    unsigned report_every_i = DTREPORT;
	unsigned dev_every_i_reports = DDREPORT;

    vector<WordIdSentences> train_src_doc, train_trg_doc;
    vector<vector<unsigned int>> train_doc_sids;

    // shuffle documents
	cerr << endl << "***SHUFFLE" << endl;
    shuffle(order.begin(), order.end(), *rndeng);

    unsigned count_steps = 0;//to count the number of updates or training steps for Adam learning rate change
	while (epoch < max_epochs) {
		transformer::ModelStats tstats;

        tf.set_dropout(true);// enable dropout

		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == train_doccor.size()) {
				//timing
				cerr << "***Epoch " << epoch << " is finished. ";
				timer_epoch.show();

				epoch++;

				id = 0;
                did = 0;
                last_print = 0;
                lines = 0;

				// learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
				if (lr_epochs > 0 && epoch >= lr_epochs)
					p_sgd->learning_rate /= lr_eta_decay; 

				if (/*step_num >= max_steps || */epoch > max_epochs) break;

				// shuffle the access order
				cerr << "***SHUFFLE" << endl;
				std::shuffle(order.begin(), order.end(), *dynet::rndeng);

				timer_epoch.reset();
			}

            // build graph for this document
            transformer::ModelStats ctstats;

            const unsigned sdlen = train_doccor[order[id % order.size()]].size();
            train_src_doc = train_src_docminibatch[order[id % order.size()]];
            train_trg_doc = train_trg_docminibatch[order[id % order.size()]];
            train_doc_sids = train_sids_docminibatch[order[id % order.size()]];
            vector<vector<vector<dynet::real>>> tsrcwordrep_doc = tsrcwordrep_cor[order[id % order.size()]];

	    	//ComputationGraph cg;

            //if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
            //    cg.set_immediate_compute(true);
            //    cg.set_check_validity(true);
            //}

            for (unsigned i = 0; i < train_src_doc.size(); ++i){
                ComputationGraph cg;

                if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
                    cg.set_immediate_compute(true);
                    cg.set_check_validity(true);
                }

                Expression i_xent = tf.build_graph(cg, train_src_doc[i], train_trg_doc[i], tsrcwordrep_doc, train_doc_sids[i], &ctstats);
                dynet::Expression& i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                float loss = dynet::as_scalar(cg.get_value(i_xent.i));
                if (!is_validloss(loss)){
                    std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
                    continue;
                }

                tstats._scores[1] += loss;

                if (PRINT_GRAPHVIZ) {
                    cerr << "***********************************************************************************" << endl;
                    cg.print_graphviz();
                    cerr << "***********************************************************************************" << endl;
                }

                cg.backward(i_objective);
				//p_sgd->update();
            }

            count_steps++;
            tstats._words_src += ctstats._words_src;
            tstats._words_src_unk += ctstats._words_src_unk;
            tstats._words_tgt += ctstats._words_tgt;
            tstats._words_tgt_unk += ctstats._words_tgt_unk;

            if (update_steps > 1){
                if (count_steps % update_steps == 0) {
					p_sgd->update();
				}
            }
            else {
				p_sgd->update();
			}

            iter++;
            did++;
            lines+=sdlen;

            if (did / report_every_i != last_print
                || iter >= dev_every_i_reports
                || id + 1 == train_sids_docminibatch.size()){
                last_print = did / report_every_i;

                float elapsed = timer_iteration.elapsed();

                p_sgd->status();
                cerr << "docs=" << did << " sents=" << lines << " ";
                cerr /*<< "loss=" << tstats._scores[1]*/ << "src_unks=" << tstats._words_src_unk << " trg_unks=" << tstats._words_tgt_unk << " " << tstats.get_score_string() << ' ';
                cerr /*<< "time_elapsed=" << elapsed*/ << "(" << (float)(tstats._words_src + tstats._words_tgt) * 1000.f / elapsed << " words/sec)" << endl;
            }
            //step_num += 1;

            //if (tfc._nlayers > 4)
            ///    p_sgd->learning_rate = (r0 / std::sqrt(tfc._num_units)) * min(1.f / std::sqrt(step_num) , step_num / pow(warmup_steps, 1.5));
            ++id;
        }

        tf.set_dropout(false);// disable dropout for evaluating dev data

        // show score on dev data?
		timer_iteration.reset();

        //eval_on_dev(tf, devel_doccor, dstats, dev_eval_mea, dev_eval_infer_algo);// non-batched version
        eval_on_dev(tf, dev_src_docminibatch, dev_trg_docminibatch, dsrcwordrep_cor, dev_sids_docminibatch, dstats,
                        dev_eval_mea, dev_eval_infer_algo);// batched version (2-3 times faster)
        float elapsed = timer_iteration.elapsed();

        // update best score and save parameter to file
        dstats.update_best_score(cpt);
        if (cpt == 0){
            // FIXME: consider average checkpointing?
            tf.save_params_to_file(params_out_file);
        }

        // verbose
        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "***DEV [epoch=" << (float)epoch + (float)id/(float)train_doccor.size() << " eta=" << p_sgd->learning_rate << "]" << " docs=" << devel_doccor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';
        if (cpt > 0) cerr << "(not improved, best score on dev so far: " << dstats.get_score_string(false) << ") ";
        cerr << "[completed in " << elapsed << " ms]" << endl;

        // learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
        if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
            cerr << "The model has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
            p_sgd->learning_rate /= lr_eta_decay;
        }

        // another early stopping criterion
        if (patience > 0 && cpt >= patience)
        {
            if (RESET_IF_STUCK){//this has not been tried out (taken from original code)
                cerr << "The model seems to get stuck. Resetting now...!" << endl;
                cerr << "Attempting to resume the training..." << endl;
                // 1) load the previous best model
                cerr << "Loading previous best model..." << endl;
                tf.initialise_params_from_file(params_out_file);
                // 2) some useful tricks:
                did = 0; id = 0; last_print = 0; cpt = 0;
                lines = 0;
                // a) reset SGD trainer, switching to Adam instead!
                if (SWITCH_TO_ADAM){
                    delete p_sgd; p_sgd = 0;
                    p_sgd = new dynet::AdamTrainer(tf.get_model_parameters(), 0.0001f/*maybe smaller?*/);
                    SWITCH_TO_ADAM = false;// do it once!
                }
                // b) use smaller batch size
                if (USE_SMALLER_MINIBATCH){
                    cerr << "Creating minibatches for training data (using minibatch_size=" << minibatch_size/2 << ")..." << endl;
                    train_src_docminibatch.clear();
                    train_trg_docminibatch.clear();
                    train_sids_docminibatch.clear();
                    create_docminibatches(train_doccor, minibatch_size/2, train_src_docminibatch, train_trg_docminibatch, train_sids_docminibatch);// on train

                    minibatch_size /= 2;
                    report_every_i /= 2;
                }
                // 3) shuffle the training data
                cerr << "***SHUFFLE" << endl;
                std::shuffle(order.begin(), order.end(), *dynet::rndeng);

                NUM_RESETS--;
                if (NUM_RESETS == 0)
                    RESET_IF_STUCK = false;// it's right time to stop anyway!
            }
            else{
                cerr << "The model has not been improved for " << patience << " times. Stopping now...!" << endl;
                cerr << "No. of epochs so far: " << epoch << "." << endl;
                cerr << "Best score on dev: " << dstats.get_score_string(false) << endl;
                cerr << "--------------------------------------------------------------------------------------------------------" << endl;

                break;
            }
        }

		timer_iteration.reset();

        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
	}

	cerr << endl << "***************************" << endl;
	cerr << "Context-based Transformer training completed!" << endl;
}
// ---

// ---
void eval_on_dev(transformer::TransformerContextModel &tf,
                 const std::vector<std::vector<WordIdSentences>> &dev_src_docminibatch, const std::vector<std::vector<WordIdSentences>> &dev_trg_docminibatch,
                 vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &dtgtwordrep_cor, std::vector<std::vector<std::vector<unsigned int>>> dev_sids_docminibatch,
                 transformer::ModelStats& dstats,
                 unsigned dev_eval_mea, unsigned dev_eval_infer_algo) // batched version of eval_on_dev
{
    vector<WordIdSentences> dev_src_doc, dev_trg_doc;
    vector<vector<unsigned int>> dev_doc_sids;

    if (dev_eval_mea == 0) // perplexity
    {
        double losses = 0.f;

        for (unsigned d = 0; d < dev_src_docminibatch.size(); ++d) {
            dev_src_doc = dev_src_docminibatch[d];
            dev_trg_doc = dev_trg_docminibatch[d];
            dev_doc_sids = dev_sids_docminibatch[d];
            vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> dtgtwordrep_doc = dtgtwordrep_cor[d];

            for (unsigned i = 0; i < dev_src_doc.size(); ++i) {
                ComputationGraph cg;

                auto i_xent = tf.build_graph(cg, dev_src_doc[i], dev_trg_doc[i], dtgtwordrep_doc, dev_doc_sids[i], nullptr, true);
                losses += as_scalar(cg.forward(i_xent));
            }
        }
        dstats._scores[1] = losses;
    }
}
// ---

// ---
void run_bilingualtrain(transformer::TransformerContextModel &tf, const DocCorpus &train_doccor, const DocCorpus &devel_doccor,
                   vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &ttgtwordrep_cor, vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &dtgtwordrep_cor,
                   dynet::Trainer*& p_sgd,
                   const std::string& params_out_file,
                   unsigned max_epochs, unsigned patience,
                   unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
                   unsigned average_checkpoints,
                   unsigned dev_eval_mea, unsigned dev_eval_infer_algo, unsigned update_steps)
{
    unsigned did = 0, id = 0, last_print = 0, lines = 0;
    MyTimer timer_epoch("completed in"), timer_iteration("completed in");
    unsigned epoch = 0, cpt = 0/*count of patience*/;

    // get current configuration
    const transformer::TransformerConfig& tfc = tf.get_config();

    // create minibatches
    std::vector<std::vector<WordIdSentences>> train_src_docminibatch, dev_src_docminibatch;
    std::vector<std::vector<WordIdSentences>> train_trg_docminibatch, dev_trg_docminibatch;
    std::vector<std::vector<std::vector<unsigned int>>> train_sids_docminibatch, dev_sids_docminibatch;
    size_t minibatch_size = MINIBATCH_SIZE;
    cerr << endl << "Creating minibatches for training data (using minibatch_size=" << minibatch_size << ")..." << endl;
    create_docminibatches(train_doccor, minibatch_size, train_src_docminibatch, train_trg_docminibatch, train_sids_docminibatch);// on train
    cerr << "Creating minibatches for development data (using minibatch_size=" << minibatch_size*2/3<< ")..." << endl;
    create_docminibatches(devel_doccor, minibatch_size*2/3, dev_src_docminibatch, dev_trg_docminibatch, dev_sids_docminibatch);// on dev
    // create a document list for the train data
    vector<unsigned> order(train_doccor.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    // model stats on dev
    transformer::ModelStats dstats(dev_eval_mea);
    get_dev_stats(devel_doccor, tfc, dstats);

    // ----
    // pre-compute model score on dev before training
    tf.set_dropout(false);// disable dropout
    //eval_on_dev(tf, devel_doccor, dstats, dev_eval_mea, dev_eval_infer_algo);// non-batched version FIXME for document-level
    eval_on_dev(tf, dev_src_docminibatch, dev_trg_docminibatch, dtgtwordrep_cor, dev_sids_docminibatch, dstats,
                dev_eval_mea, dev_eval_infer_algo);// batched version (2-3 times faster) FIXME for document-level
    float elapsed = timer_iteration.elapsed();
    timer_iteration.reset();

    // update best score so far (in case of incremental training)
    dstats.update_best_score(cpt);

    // verbose
    cerr << endl << "--------------------------------------------------------------------------------------------------------" << endl;
    cerr << "***Initial score on DEV: docs=" << devel_doccor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';
    cerr << "[completed in " << elapsed << " ms]" << endl;
    cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    // ----

    unsigned report_every_i = DTREPORT;
    unsigned dev_every_i_reports = DDREPORT;

    vector<WordIdSentences> train_src_doc, train_trg_doc;
    vector<vector<unsigned int>> train_doc_sids;

    // shuffle documents
    cerr << endl << "***SHUFFLE" << endl;
    shuffle(order.begin(), order.end(), *rndeng);

    unsigned count_steps = 0;//to count the number of updates or training steps for Adam learning rate change
    while (epoch < max_epochs) {
        transformer::ModelStats tstats;

        tf.set_dropout(true);// enable dropout

        for (unsigned iter = 0; iter < dev_every_i_reports;) {
            if (id == train_doccor.size()) {
                //timing
                cerr << "***Epoch " << epoch << " is finished. ";
                timer_epoch.show();

                epoch++;

                id = 0;
                did = 0;
                last_print = 0;
                lines = 0;

                // learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
                if (lr_epochs > 0 && epoch >= lr_epochs)
                    p_sgd->learning_rate /= lr_eta_decay;

                if (/*step_num >= max_steps || */epoch > max_epochs) break;

                // shuffle the access order
                cerr << "***SHUFFLE" << endl;
                std::shuffle(order.begin(), order.end(), *dynet::rndeng);

                timer_epoch.reset();
            }

            // build graph for this document
            transformer::ModelStats ctstats;

            const unsigned sdlen = train_doccor[order[id % order.size()]].size();
            train_src_doc = train_src_docminibatch[order[id % order.size()]];
            train_trg_doc = train_trg_docminibatch[order[id % order.size()]];
            train_doc_sids = train_sids_docminibatch[order[id % order.size()]];
            vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> ttgtwordrep_doc = ttgtwordrep_cor[order[id % order.size()]];

            //ComputationGraph cg;

            //if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
            //    cg.set_immediate_compute(true);
            //    cg.set_check_validity(true);
            //}

            for (unsigned i = 0; i < train_src_doc.size(); ++i){
                ComputationGraph cg;

                if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
                    cg.set_immediate_compute(true);
                    cg.set_check_validity(true);
                }

                Expression i_xent = tf.build_graph(cg, train_src_doc[i], train_trg_doc[i], ttgtwordrep_doc, train_doc_sids[i], &ctstats);
                dynet::Expression& i_objective = i_xent;

                // perform forward computation for aggregate objective
                cg.forward(i_objective);

                // grab the parts of the objective
                float loss = dynet::as_scalar(cg.get_value(i_xent.i));
                if (!is_validloss(loss)){
                    std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
                    continue;
                }

                tstats._scores[1] += loss;

                if (PRINT_GRAPHVIZ) {
                    cerr << "***********************************************************************************" << endl;
                    cg.print_graphviz();
                    cerr << "***********************************************************************************" << endl;
                }

                cg.backward(i_objective);
                //p_sgd->update();

            }

            count_steps++;
            tstats._words_src += ctstats._words_src;
            tstats._words_src_unk += ctstats._words_src_unk;
            tstats._words_tgt += ctstats._words_tgt;
            tstats._words_tgt_unk += ctstats._words_tgt_unk;

			if (update_steps > 1){
				if (count_steps % update_steps == 0) {
					p_sgd->update();
				}
			}
			else {
				p_sgd->update();
			}

			iter++;
            did++;
            lines+=sdlen;

            if (did / report_every_i != last_print
                || iter >= dev_every_i_reports
                || id + 1 == train_sids_docminibatch.size()){
                last_print = did / report_every_i;

                float elapsed = timer_iteration.elapsed();

                p_sgd->status();
                cerr << "docs=" << did << " sents=" << lines << " ";
                cerr /*<< "loss=" << tstats._scores[1]*/ << "src_unks=" << tstats._words_src_unk << " trg_unks=" << tstats._words_tgt_unk << " " << tstats.get_score_string() << ' ';
                cerr /*<< "time_elapsed=" << elapsed*/ << "(" << (float)(tstats._words_src + tstats._words_tgt) * 1000.f / elapsed << " words/sec)" << endl;
            }
            //step_num += 1;

            //if (tfc._nlayers > 4)
            ///    p_sgd->learning_rate = (r0 / std::sqrt(tfc._num_units)) * min(1.f / std::sqrt(step_num) , step_num / pow(warmup_steps, 1.5));
            ++id;
        }

        tf.set_dropout(false);// disable dropout for evaluating dev data

        // show score on dev data?
        timer_iteration.reset();

        //eval_on_dev(tf, devel_doccor, dstats, dev_eval_mea, dev_eval_infer_algo);// non-batched version
        eval_on_dev(tf, dev_src_docminibatch, dev_trg_docminibatch, dtgtwordrep_cor, dev_sids_docminibatch, dstats,
                    dev_eval_mea, dev_eval_infer_algo);// batched version (2-3 times faster)
        float elapsed = timer_iteration.elapsed();

        // update best score and save parameter to file
        dstats.update_best_score(cpt);
        if (cpt == 0){
            // FIXME: consider average checkpointing?
            tf.save_params_to_file(params_out_file);
        }

        // verbose
        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "***DEV [epoch=" << (float)epoch + (float)id/(float)train_doccor.size() << " eta=" << p_sgd->learning_rate << "]" << " docs=" << devel_doccor.size() << " src_unks=" << dstats._words_src_unk << " trg_unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';
        if (cpt > 0) cerr << "(not improved, best score on dev so far: " << dstats.get_score_string(false) << ") ";
        cerr << "[completed in " << elapsed << " ms]" << endl;

        // learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
        if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
            cerr << "The model has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
            p_sgd->learning_rate /= lr_eta_decay;
        }

        // another early stopping criterion
        if (patience > 0 && cpt >= patience)
        {
            if (RESET_IF_STUCK){//not tried out
                cerr << "The model seems to get stuck. Resetting now...!" << endl;
                cerr << "Attempting to resume the training..." << endl;
                // 1) load the previous best model
                cerr << "Loading previous best model..." << endl;
                tf.initialise_params_from_file(params_out_file);
                // 2) some useful tricks:
                did = 0; id = 0; last_print = 0; cpt = 0;
                lines = 0;
                // a) reset SGD trainer, switching to Adam instead!
                if (SWITCH_TO_ADAM){
                    delete p_sgd; p_sgd = 0;
                    p_sgd = new dynet::AdamTrainer(tf.get_model_parameters(), 0.0001f/*maybe smaller?*/);
                    SWITCH_TO_ADAM = false;// do it once!
                }
                // b) use smaller batch size
                if (USE_SMALLER_MINIBATCH){
                    cerr << "Creating minibatches for training data (using minibatch_size=" << minibatch_size/2 << ")..." << endl;
                    train_src_docminibatch.clear();
                    train_trg_docminibatch.clear();
                    train_sids_docminibatch.clear();
                    create_docminibatches(train_doccor, minibatch_size/2, train_src_docminibatch, train_trg_docminibatch, train_sids_docminibatch);// on train

                    minibatch_size /= 2;
                    report_every_i /= 2;
                }
                // 3) shuffle the training data
                cerr << "***SHUFFLE" << endl;
                std::shuffle(order.begin(), order.end(), *dynet::rndeng);

                NUM_RESETS--;
                if (NUM_RESETS == 0)
                    RESET_IF_STUCK = false;// it's right time to stop anyway!
            }
            else{
                cerr << "The model has not been improved for " << patience << " times. Stopping now...!" << endl;
                cerr << "No. of epochs so far: " << epoch << "." << endl;
                cerr << "Best score on dev: " << dstats.get_score_string(false) << endl;
                cerr << "--------------------------------------------------------------------------------------------------------" << endl;

                break;
            }
        }

        timer_iteration.reset();

        cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    }

    cerr << endl << "***************************" << endl;
    cerr << "Context-based Transformer training completed!" << endl;
}
// ---

//---
std::string get_sentence(const WordIdSentence& source, Dict& td){
	WordId eos_sym = td.convert("</s>");

	std::stringstream ss;
	for (WordId w : source){
		if (w == eos_sym) {
			ss << "</s>";
			break;// stop if seeing EOS marker!
		}

		ss << td.convert(w) << " ";
	}

	return ss.str();
}
//---

//---

void save_config(const std::string& config_out_file, const std::string& params_out_file, const TransformerConfig& tfc)
{
	// line has the format:
	// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <online-docmt> <doc-attention-type> <context-type> <use-sparse-soft4hier> <your-trained-model-path>
	// e.g.,
	// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 0 1 1 1 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010101_att1_ls01_pe1_ml300_ffrelu_run1
	std::stringstream ss;
		
	ss << tfc._num_units << " " << tfc._nheads << " " << tfc._nlayers << " " << tfc._n_ff_units_factor << " "
		<< tfc._encoder_emb_dropout_rate << " " << tfc._encoder_sublayer_dropout_rate << " " << tfc._decoder_emb_dropout_rate << " " << tfc._decoder_sublayer_dropout_rate << " " << tfc._attention_dropout_rate << " " << tfc._ff_dropout_rate << " "
		<< tfc._use_label_smoothing << " " << tfc._label_smoothing_weight << " "
		<< tfc._position_encoding << " " << tfc._position_encoding_flag << " " << tfc._max_length << " "
		<< tfc._attention_type << " "
		<< tfc._ffl_activation_type << " "
		<< tfc._shared_embeddings << " "
		<< tfc._use_hybrid_model << " "
        << tfc._online_docmt << " "
        << tfc._doc_attention_type << " "
        << tfc._context_type << " "
        << tfc._use_sparse_soft << " ";
    ss << params_out_file;

	ofstream outf_cfg(config_out_file);
	assert(outf_cfg);
	outf_cfg << ss.str();
}
//---