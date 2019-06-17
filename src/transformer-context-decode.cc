/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
* Extended for Context-based NMT by Sameen Maruf
*/

#include "transformer-context.h"

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
bool load_model_config(const std::string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm);
// ---

void load_wordrep(vector<vector<vector<vector<dynet::real>>>> &sent_rep, const string &rep_file);
void load_tgtrep(vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> &sent_rep, const string &rep_file);

// ---
void greedy(const std::string& test_file
		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
		, unsigned length_ratio=2.f
		, bool remove_unk=false /*whether to include <unk> in the output*/
		, bool r2l_target=false /*right-to-left decoding*/);
void greedy_monolingual(const std::string& test_file
		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
		, unsigned length_ratio=2.f
		, bool remove_unk=false /*whether to include <unk> in the output*/
		, bool r2l_target=false /*right-to-left decoding*/);
void greedy_bilingual(const std::string& test_file
		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
		, unsigned length_ratio=2.f
		, bool remove_unk=false /*whether to include <unk> in the output*/
		, bool r2l_target=false /*right-to-left decoding*/);
// ---
//void rescore_monolingual(const std::string& test_file
//		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
//		, bool r2l_target=false /*right-to-left decoding*/);
//void rescore_bilingual(const std::string& test_file
//		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
//      , unsigned length_ratio=2.f
//      , bool r2l_target=false /*right-to-left decoding*/);
//---

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
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("model-path,p", value<std::string>()->default_value("."), "specify pre-trained model path")
		("model-file", value<std::string>()->default_value("."), "specify pre-trained model file from which to load")
		//-----------------------------------------
		("test,T", value<std::string>(), "file containing testing sentences (should be docid ||| source for greedy decoding)")
		("lc", value<unsigned int>()->default_value(0), "specify the sentence/line number to be continued (for decoding only); 0 by default")
		("context-type", value<unsigned>()->default_value(1), "decide which type of context to use (0: decode without context, 1: monolingual from encoder, 2: bilingual from decoder)")
		//-----------------------------------------
		("beam,b", value<unsigned>()->default_value(1), "size of beam in decoding; 1: greedy by default")
		//("rescore", "if you want to score the target instead of generating it")
		//("alpha,a", value<float>()->default_value(0.6f), "length normalisation hyperparameter; 0.6f by default") // follow the GNMT paper!
		//("topk,k", value<unsigned>(), "use <num> top kbest entries; none by default")
		//("nbest-style", value<std::string>()->default_value("simple"), "style for nbest translation outputs (moses|simple); simple by default")
		("length-ratio", value<unsigned>()->default_value(2), "target_length = source_length * TARGET_LENGTH_LIMIT_FACTOR; 2 by default")
		//-----------------------------------------
		//-----------------------------------------
		("remove-unk", "remove <unk> in the output; default not")
		//-----------------------------------------
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
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
		|| !(vm.count("model-path") || !vm.count("test")))
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
	std::vector<std::shared_ptr<transformer::TransformerContextModel>> v_tf_models;

    std::string config_file = model_path + "/" + vm["model-file"].as<std::string>() + ".config";
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

		if (vm.count("swap")){
			std::swap(sd, td);
			std::swap(sm._kSRC_SOS, sm._kTGT_SOS);
			std::swap(sm._kSRC_EOS, sm._kTGT_EOS);
			std::swap(sm._kSRC_UNK, sm._kTGT_UNK);
		}

		// load models
		if (!load_model_config(config_file, v_tf_models, sd, td, sm))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s)!");
	}
	else TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s) from: " + std::string(model_path) + "!");

    // input test file
	// the output will be printed to stdout!
	std::string test_input_file = vm["test"].as<std::string>();

	// decode the input file
	/*
	if (vm.count("rescore")){
		if (vm["context-type"].as<unsigned>() == 1)
			rescore_monolingual(test_input_file, v_tf_models, vm.count("r2l-target"));
		else
			rescore_bilingual(test_input_file, v_tf_models, vm.count("r2l-target"));
	}
	else {
	 */
	if (vm["context-type"].as<unsigned>() == 0) {
		if (vm["beam"].as<unsigned>() == 1)
			greedy(test_input_file, v_tf_models, vm["length-ratio"].as<unsigned>(),
				   vm.count("remove-unk"), vm.count("r2l-target"));
		else
			TRANSFORMER_RUNTIME_ASSERT("Beam-size should always be 1!");
	} else if (vm["context-type"].as<unsigned>() == 1) {
		if (vm["beam"].as<unsigned>() == 1)
			greedy_monolingual(test_input_file, v_tf_models, vm["length-ratio"].as<unsigned>(),
							   vm.count("remove-unk"), vm.count("r2l-target"));
		else
			TRANSFORMER_RUNTIME_ASSERT("Beam-size should always be 1!");
	} else {
		if (vm["beam"].as<unsigned>() == 1)
			greedy_bilingual(test_input_file, v_tf_models, vm["length-ratio"].as<unsigned>(),
							 vm.count("remove-unk"), vm.count("r2l-target"));
		else
			TRANSFORMER_RUNTIME_ASSERT("Beam-size should always be 1!");
	}
	//}
    return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
bool load_model_config(const std::string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
	, dynet::Dict& sd
	, dynet::Dict& td
	, const transformer::SentinelMarkers& sm)
{
	cerr << "Loading model(s) from configuration file: " << model_cfg_file << "..." << endl;	

	v_models.clear();

	ifstream inpf(model_cfg_file);
	assert(inpf);
	
	unsigned i = 0;
	std::string line;
	while (getline(inpf, line)){
		if ("" == line) break;

		// each line has the format: 
		// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <encoder-emb-dropout> <encoder-sub-layer-dropout> <decoder-emb-dropout> <decoder-sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <position-encoding-flag> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <online-docmt> <doc-attention-type> <context-type> <use-sparse-soft4hier> <your-trained-model-path>
		// e.g.,
		// 128 2 2 4 0.1 0.1 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 0 1 1 1 <your-path>/models/iwslt-envi/params.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml300_ffrelu_run1
		cerr << "Loading model " << i+1 << "..." << endl;
		std::stringstream ss(line);

		transformer::TransformerConfig tfc;
        std::string model_file;

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
        ss >> model_file;
        tfc._is_training = false;
		tfc._use_dropout = false;

		v_models.push_back(std::shared_ptr<transformer::TransformerContextModel>());
		v_models[i].reset(new transformer::TransformerContextModel(tfc, sd, td));

        cerr << "Model file: " << model_file << endl;
		v_models[i].get()->initialise_params_from_file(model_file);// load pre-trained model from file
		cerr << "Count of model parameters: " << v_models[i].get()->get_model_parameters().parameter_count() << endl;

		i++;
	}

	cerr << "Done!" << endl << endl;

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
void greedy(const std::string& test_file
		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
		, unsigned length_ratio
		, bool remove_unk /*whether to include <unk> in the output*/
		, bool r2l_target /*right-to-left decoding*/)
{
	dynet::Dict& sd = v_models[0].get()->get_source_dict();
	dynet::Dict& td = v_models[0].get()->get_target_dict();
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;

	SrcCorpus sent_corpus;
	SrcDocCorpus doc_corpus;
	cerr << "Reading test examples from " << test_file << endl;
	bool use_joint_vocab = v_models[0].get()->get_config()._shared_embeddings;
	if (use_joint_vocab){
		sent_corpus = read_srcdoccorpus(test_file, &sd, &sd);
		doc_corpus = read_srcdoccorpus(sent_corpus);
		td = sd;
	}
	else {
		sent_corpus = read_srcdoccorpus(test_file, &sd, &td);
		doc_corpus = read_srcdoccorpus(sent_corpus);
	}

	MyTimer timer_dec("completed in");
	WordIdSentences vssent;
	unsigned int lno = 0;
	for (unsigned i = 0; i < doc_corpus.size(); ++i) {
		cout << "<d>" << endl;
		const unsigned tdlen = doc_corpus[i].size();

		for (unsigned dl = 0; dl < tdlen; ++dl)
			vssent.push_back(doc_corpus[i].at(dl));

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;// dynamic computation graph
			WordIdSentence source = vssent[dl];
			WordIdSentence target = v_models[0].get()->greedy_decode(cg, source, length_ratio);

			if (r2l_target)
				std::reverse(target.begin() + 1, target.end() - 1);

			bool first = true;
			for (auto &w: target) {
				if (!first) cout << " ";

				if (remove_unk && w == sm._kTGT_UNK) continue;

				cout << td.convert(w);

				first = false;
			}
			cout << endl;

			lno++;
			//break;//for debug only
		}
		vssent.clear();
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Greedy decoding is finished!" << endl;
	cerr << "Decoded " << lno << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}
// ---

// ---
void greedy_monolingual(const std::string& test_file
		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
		, unsigned length_ratio
		, bool remove_unk /*whether to include <unk> in the output*/
		, bool r2l_target /*right-to-left decoding*/)
{
	dynet::Dict& sd = v_models[0].get()->get_source_dict();
	dynet::Dict& td = v_models[0].get()->get_target_dict();
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;

    SrcCorpus sent_corpus;
    SrcDocCorpus doc_corpus;
	cerr << "Reading test examples from " << test_file << endl;
    bool use_joint_vocab = v_models[0].get()->get_config()._shared_embeddings;
    if (use_joint_vocab){
        sent_corpus = read_srcdoccorpus(test_file, &sd, &sd);
        doc_corpus = read_srcdoccorpus(sent_corpus);
        td = sd;
    }
    else {
        sent_corpus = read_srcdoccorpus(test_file, &sd, &td);
        doc_corpus = read_srcdoccorpus(sent_corpus);
    }

	//compute the required representations
	vector<vector<vector<vector<dynet::real>>>> tsrcwordrep_cor;
	for (unsigned sd = 0; sd < doc_corpus.size(); sd++) {
		vector<vector<vector<dynet::real>>> srcwordrep_doc;

		const unsigned sdlen = doc_corpus[sd].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) {
			ComputationGraph cg;
			WordIdSentence source = doc_corpus[sd].at(dl);
			srcwordrep_doc.push_back(v_models[0].get()->compute_source_rep(cg, source));
		}
		tsrcwordrep_cor.push_back(srcwordrep_doc);
	}
	cerr << "Computed representations of " << doc_corpus.size() << " documents!" << endl;


	MyTimer timer_dec("completed in");
	WordIdSentences vssent;
	unsigned int lno = 0;
    for (unsigned i = 0; i < doc_corpus.size(); ++i) {
        cout << "<d>" << endl;
        const unsigned tdlen = doc_corpus[i].size();
        vector<unsigned int> sids;
		vector<vector<vector<dynet::real>>> tsrcwordrep_doc = tsrcwordrep_cor[i];

        for (unsigned dl = 0; dl < tdlen; ++dl)
            vssent.push_back(doc_corpus[i].at(dl));

        for (unsigned dl = 0; dl < vssent.size(); ++dl) {
        	//cout << endl;
			//cout << "Sentence " << dl << endl;
            ComputationGraph cg;// dynamic computation graph
            WordIdSentence source = vssent[dl];
            sids.push_back(dl);
            WordIdSentence target = v_models[0].get()->greedy_decode(cg, source, tsrcwordrep_doc, sids, length_ratio);

            if (r2l_target)
                std::reverse(target.begin() + 1, target.end() - 1);

            bool first = true;
            for (auto &w: target) {
                if (!first) cout << " ";

                if (remove_unk && w == sm._kTGT_UNK) continue;

                cout << td.convert(w);

                first = false;
            }
            cout << endl;

            lno++;
            sids.clear();
            //break;//for debug only
        }
        vssent.clear();
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Greedy decoding Transformer with Monolingual Context is finished!" << endl;
	cerr << "Decoded " << lno << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}

void greedy_bilingual(const std::string& test_file
		, std::vector<std::shared_ptr<transformer::TransformerContextModel>>& v_models
		, unsigned length_ratio
		, bool remove_unk /*whether to include <unk> in the output*/
		, bool r2l_target /*right-to-left decoding*/)
{
	dynet::Dict& sd = v_models[0].get()->get_source_dict();
	dynet::Dict& td = v_models[0].get()->get_target_dict();
	const transformer::SentinelMarkers& sm = v_models[0].get()->get_config()._sm;

	SrcCorpus sent_corpus;
	SrcDocCorpus doc_corpus;
	cerr << "Reading test examples from " << test_file << endl;
	bool use_joint_vocab = v_models[0].get()->get_config()._shared_embeddings;
	if (use_joint_vocab){
		sent_corpus = read_srcdoccorpus(test_file, &sd, &sd);
		doc_corpus = read_srcdoccorpus(sent_corpus);
		td = sd;
	}
	else {
		sent_corpus = read_srcdoccorpus(test_file, &sd, &td);
		doc_corpus = read_srcdoccorpus(sent_corpus);
	}

	//compute the required representations here
	vector<vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>>> ttgtwordrep_cor;
	for (unsigned sd = 0; sd < doc_corpus.size(); sd++) {
		vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc;

		const unsigned sdlen = doc_corpus[sd].size();
		for (unsigned dl = 0; dl < sdlen; ++dl) {
			ComputationGraph cg;
			WordIdSentence source = doc_corpus[sd].at(dl);
			tgtwordrep_doc.push_back(v_models[0].get()->compute_bilingual_rep(cg, source, length_ratio));
		}
		ttgtwordrep_cor.push_back(tgtwordrep_doc);
	}
	cerr << "Computed representations of " << doc_corpus.size() << " documents!" << endl;

	MyTimer timer_dec("completed in");
	WordIdSentences vssent;
	unsigned int lno = 0;
	for (unsigned i = 0; i < doc_corpus.size(); ++i) {
		cout << "<d>" << endl;
		const unsigned tdlen = doc_corpus[i].size();
		vector<unsigned int> sids;
		vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> ttgtwordrep_doc = ttgtwordrep_cor[i];

		for (unsigned dl = 0; dl < tdlen; ++dl)
			vssent.push_back(doc_corpus[i].at(dl));

		for (unsigned dl = 0; dl < vssent.size(); ++dl) {
			ComputationGraph cg;// dynamic computation graph
			WordIdSentence source = vssent[dl];
			sids.push_back(dl);
			WordIdSentence target = v_models[0].get()->greedy_decode(cg, source, ttgtwordrep_doc, sids, length_ratio);

			if (r2l_target)
				std::reverse(target.begin() + 1, target.end() - 1);

			bool first = true;
			for (auto &w: target) {
				if (!first) cout << " ";

				if (remove_unk && w == sm._kTGT_UNK) continue;

				cout << td.convert(w);

				first = false;
			}
			cout << endl;

			lno++;
			sids.clear();
			//break;//for debug only
		}
		vssent.clear();
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Greedy decoding for Transformer with Bilingual Context is finished!" << endl;
	cerr << "Decoded " << lno << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}
// ---

