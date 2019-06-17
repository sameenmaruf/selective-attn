/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang and Extended by Sameen Maruf
* Created: 11 Sep 2018
* This file has the functions for computing representations from base Transformer model
 * Created by Sameen Maruf
*/

#include "transformer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <sys/stat.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace boost::program_options;

// ---
bool load_data(const string& data_file, SrcDocCorpus& doc_corpus
		, dynet::Dict& sd, dynet::Dict& td, bool use_joint_vocab
		, SentinelMarkers& sm);

bool load_model_config(const std::string& model_path
	, std::shared_ptr<transformer::TransformerModel>& model
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm);
// ---
void GetSrcRep(std::shared_ptr<transformer::TransformerModel>& model, SrcDocCorpus &src_doccorpus, const string& out_file);
void GetTgtRep(std::shared_ptr<transformer::TransformerModel>& model, SrcDocCorpus &src_doccorpus, const string& out_file, unsigned length_ratio);
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
		("model-path,p", value<std::string>()->default_value("."), "specify pre-trained model path")
		//-----------------------------------------
		("input_doc", value<std::string>(), "file containing sentences, with each line consisting of docID ||| source.")
		("input_type", value<unsigned>()->default_value(0), "decide which source file representations are to be computed (0: train, 1: dev, 2:test)")
		("lc", value<unsigned int>()->default_value(0), "specify the sentence/line number to be continued (for decoding only); 0 by default")
		//-----------------------------------------
		("rep_type", value<unsigned>()->default_value(1), "decide which representations are to be computed (1: monolingual from encoder, 2: bilingual from decoder)")
		("length-ratio", value<unsigned>()->default_value(2), "target_length = source_length * TARGET_LENGTH_LIMIT_FACTOR; 2 by default")
		//-----------------------------------------
		//("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		("dynet-profiling", value<int>()->default_value(0), "enable/disable auto profiling (https://github.com/clab/dynet/pull/1088/commits/bc34db98fa5e2e694f54f0e6b1d720d517c7530e)")// for debugging only		
		//-----------------------------------------
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
		|| !(vm.count("model-path") || !vm.count("input_doc")))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// get and check model path
	std::string model_path = vm["model-path"].as<std::string>();
	struct stat sb;
	if (stat(model_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		cerr << endl << "All model files will be loaded from: " << model_path << "." << endl;
	else
		TRANSFORMER_RUNTIME_ASSERT("The model-path does not exist!");

	// Model recipe
	dynet::Dict sd, td;// vocabularies
	SentinelMarkers sm;// sentinel markers
	std::shared_ptr<transformer::TransformerModel> tf_model;

	std::string config_file = model_path + "/base-model.config";// configuration file path
	if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence	
		// load vocabulary from file(s)
		std::string vocab_file = model_path + "/" + "src-tgt.joint-vocab";
		if (stat(vocab_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){
			load_vocab(vocab_file, sd);
			td = sd;
		}
		else{
			std::string src_vocab_file = model_path + "/" + "src.vocab";
			std::string tgt_vocab_file = model_path + "/" + "tgt.vocab";
			load_vocabs(src_vocab_file, tgt_vocab_file, sd, td);
		}

		transformer::SentinelMarkers sm;
		sm._kSRC_SOS = sd.convert("<s>");
		sm._kSRC_EOS = sd.convert("</s>");
		sm._kSRC_UNK = sd.convert("<unk>");
		sm._kTGT_SOS = td.convert("<s>");
		sm._kTGT_EOS = td.convert("</s>");
		sm._kTGT_UNK = td.convert("<unk>");

		// load models
		if (!load_model_config(model_path, tf_model, sd, td, sm))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s)!");
	}
	else TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s) from: " + std::string(model_path) + "!");

	SrcDocCorpus doc_cor;// document-level corpus
    bool use_joint_vocab = tf_model.get()->get_config()._shared_embeddings;
    std::string data_path = vm["input_doc"].as<std::string>();// single training data file
    if (!load_data(data_path, doc_cor, sd, td, use_joint_vocab, sm))
		TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

	// input test file
    unsigned rep_type = vm["rep_type"].as<unsigned>();
    unsigned inp_type = vm["input_type"].as<unsigned>();

	//set the appropriate path for saving the representations
    std::string output_file;
    if (rep_type == 1) {
        cerr << "Computing the monolingual source representations..." << endl;
        if (inp_type == 0)
			output_file = model_path + "/src-rep/trainrep.bin";
		else if (inp_type == 1)
			output_file = model_path + "/src-rep/devrep.bin";
		else
			output_file = model_path + "/src-rep/testrep.bin";

		// compute the source representations from the encoder
		GetSrcRep(tf_model, doc_cor, output_file);
	}
	else {
		cerr << "Computing the bilingual source/target representations..." << endl;
		if (inp_type == 0)
			output_file = model_path + "/tgt-rep/trainrep.bin";
		else if (inp_type == 1)
			output_file = model_path + "/tgt-rep/devrep.bin";
		else
			output_file = model_path + "/tgt-rep/testrep.bin";

		// compute the source representations from the encoder
		GetTgtRep(tf_model, doc_cor, output_file, vm["length-ratio"].as<unsigned>());
	}

	cerr << "The representations have been saved in " << output_file << endl;

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************
//---

void GetSrcRep(std::shared_ptr<transformer::TransformerModel>& model, SrcDocCorpus &src_doccorpus, const string& out_file)
{
	vector<vector<vector<vector<dynet::real>>>> srcwordrep_corpus;
	WordIdSentence source;
	unsigned report = 0;

	for (unsigned sd = 0; sd < src_doccorpus.size(); sd++) {
		vector<vector<vector<dynet::real>>> srcwordrep_doc;

		const unsigned sdlen = src_doccorpus[sd].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) {
			ComputationGraph cg;
			source = src_doccorpus[sd].at(dl);
			srcwordrep_doc.push_back(model->compute_source_rep(cg, source));
		}
		srcwordrep_corpus.push_back(srcwordrep_doc);

		report += 1;

		if (report%50==0)
			cerr << "Computed representations of " << report << " documents so far..." << endl;
	}
	cerr << "Done with computing representations of " << src_doccorpus.size() << " documents!" << endl;

	ofstream out(out_file);
	boost::archive::binary_oarchive oa(out);
	oa << srcwordrep_corpus;
	out.close();
}

void GetTgtRep(std::shared_ptr<transformer::TransformerModel>& model, SrcDocCorpus &src_doccorpus, const string& out_file, unsigned length_ratio)
{
	vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> tgtwordrep_corpus;
	WordIdSentence source;
	unsigned report = 0;

	for (unsigned sd = 0; sd < src_doccorpus.size(); sd++) {
		vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc;

		const unsigned sdlen = src_doccorpus[sd].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) {
			ComputationGraph cg;
			source = src_doccorpus[sd].at(dl);
			tgtwordrep_doc.push_back(model->compute_bilingual_rep(cg, source, length_ratio));
		}
		tgtwordrep_corpus.push_back(tgtwordrep_doc);

		report += 1;

		if (report%50==0)
			cerr << "Computed representations of " << report << " documents so far..." << endl;
	}
	cerr << "Done with computing representations of " << src_doccorpus.size() << " documents!" << endl;

	ofstream out(out_file);
	boost::archive::binary_oarchive oa(out);
	oa << tgtwordrep_corpus;
	out.close();
}
//---

// ---
bool load_data(const string& data_file, SrcDocCorpus& doc_corpus
		, dynet::Dict& sd, dynet::Dict& td, bool use_joint_vocab
		, SentinelMarkers& sm)
{
	SrcCorpus sent_corpus;
	cerr << "Reading data from " << data_file << "...\n";
	if (use_joint_vocab){
		sent_corpus = read_srcdoccorpus(data_file, &sd, &sd);
		doc_corpus = read_srcdoccorpus(sent_corpus);
		td = sd;
	}
	else {
		sent_corpus = read_srcdoccorpus(data_file, &sd, &td);
		doc_corpus = read_srcdoccorpus(sent_corpus);
	}

	// set up <unk> ids
	sd.set_unk("<unk>");
	sm._kSRC_UNK = sd.get_unk_id();
	td.set_unk("<unk>");
	sm._kTGT_UNK = td.get_unk_id();

	return true;
}
// ---

// ---
bool load_model_config(const std::string& model_path
	, std::shared_ptr<transformer::TransformerModel>& model
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm)
{
	std::string model_cfg_file = model_path + "/base-model.config";// configuration file path

	cerr << "Loading model(s) from configuration file: " << model_cfg_file << "..." << endl;

	ifstream inpf(model_cfg_file);
	assert(inpf);
	
	std::string line;
	while (getline(inpf, line)){
		if ("" == line) break;

		// format:
		// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <position-encoding-flag> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model>
		// e.g.,
		// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0
		cerr << "Loading model ..." << endl;
		std::stringstream ss(line);

		transformer::TransformerConfig tfc;
        std::string model_file = model_path + "/base-model.params";// param file path

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
		   >> tfc._use_hybrid_model;		
		tfc._is_training = false;
		tfc._use_dropout = false;

		model.reset(new transformer::TransformerModel(tfc, sd, td));
		cerr << "Model file: " << model_file << endl;
		model.get()->initialise_params_from_file(model_file);// load pre-trained model from file
		cerr << "Count of model parameters: " << model.get()->get_model_parameters().parameter_count() << endl;
	}

	cerr << "Done!" << endl << endl;

	return true;
}
// ---

// ---

