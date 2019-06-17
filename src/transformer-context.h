/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
* Extended for Context-based NMT by Sameen Maruf
*/

#pragma once

// All utilities
#include "utils.h"

// Layers
#include "layers.h"

// Base Transformer
#include "transformer.h"

using namespace std;
using namespace dynet;

namespace transformer {

//--- Encoder Context Layer
struct EncoderContext{
    explicit EncoderContext(DyNetModel* mod, TransformerConfig& tfc, Encoder* p_encoder)
            : _context_attention_sublayer(mod, tfc)
            , _context_hierattention_sublayer(mod, tfc)
            , _feed_forward_sublayer(mod, tfc)
    {
        // for layer normalisation
        _p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
        _p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
        _p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
        _p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

        // for context gating
        _p_Cs = mod->add_parameters({tfc._num_units, tfc._num_units});
        _p_Csc = mod->add_parameters({tfc._num_units, tfc._num_units});

        _p_tfc = &tfc;

        _p_encoder = p_encoder;
    }

    ~EncoderContext(){}

    //for masking the current/future sentence/s
    MaskSent _sent_mask;
    MaskSent _word_mask;

    // multi-head attention sub-layer for the sent/word-level context
    MultiHeadContextAttentionLayer _context_attention_sublayer;

    // multi-head attention sub-layer for the hierarchical context
    MultiHeadContextHierarchicalAttentionLayer _context_hierattention_sublayer;

    // position-wise feed forward sub-layer
    FeedForwardLayer _feed_forward_sublayer;

    // for layer normalisation
    dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
    dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

    // for context gating
    dynet::Parameter _p_Cs, _p_Csc;

    // transformer config pointer
    TransformerConfig* _p_tfc = nullptr;

    // encoder object pointer
    Encoder* _p_encoder = nullptr;

    dynet::Expression compute_sentrep_and_masks(dynet::ComputationGraph &cg
            , vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids)
    {
        unsigned bsize = sids.size();

        std::vector<std::vector<float>> v_seq_masks(bsize);
        vector<Expression> sent_rep;
        for (unsigned i = 0; i < srcwordrep_doc.size(); i++) {
            vector<vector<dynet::real>> wordrep_sent = srcwordrep_doc[i];
            vector<Expression> word_rep;

            for (unsigned j = 1; j < wordrep_sent.size() - 1; ++j)//no need to have the <bos> and <eos> tokens in the cache
                word_rep.push_back(dynet::input(cg, {_p_tfc->_num_units}, wordrep_sent[j]));

            sent_rep.push_back(average(word_rep));
        }
        dynet::Expression i_sent_ctx = dynet::concatenate_to_batch(std::vector<dynet::Expression>(bsize, dynet::concatenate_cols(sent_rep)));//((num_units, ds), batch_size)

        for (unsigned l = 0; l < srcwordrep_doc.size(); l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                if (!_p_tfc->_online_docmt) {//masking for offline document MT
                    if (sids[bs] != l)//pad the current sentence index
                        v_seq_masks[bs].push_back(0.f);// padding position
                    else
                        v_seq_masks[bs].push_back(1.f);// padding position
                }
                else {//masking for online document MT
                    if (sids[bs] > l)//pad the current and future sentence index
                        v_seq_masks[bs].push_back(0.f);// padding position
                    else
                        v_seq_masks[bs].push_back(1.f);// padding position
                }
            }
        }

        _sent_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _sent_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _sent_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif

        return i_sent_ctx;
    }

    dynet::Expression compute_wordrep_and_masks(dynet::ComputationGraph &cg
            , vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, vector<unsigned int>& sent_size)
    {
        unsigned bsize = sids.size();

        std::vector<std::vector<float>> v_seq_masks(bsize);
        vector<Expression> word_rep;
        for (unsigned i = 0; i < srcwordrep_doc.size(); i++) {
            vector<vector<dynet::real>> wordrep_sent = srcwordrep_doc[i];

            for (unsigned j = 1; j < wordrep_sent.size() - 1; ++j)//no need to have the <bos> and <eos> tokens in the cache
                word_rep.push_back(dynet::input(cg, {_p_tfc->_num_units}, wordrep_sent[j]));

            sent_size.push_back(wordrep_sent.size() - 2);
        }
        dynet::Expression i_sent_ctx = dynet::concatenate_to_batch(std::vector<dynet::Expression>(bsize, dynet::concatenate_cols(word_rep)));//((num_units, dw), batch_size)

        if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::WORD) {
            for (unsigned l = 0; l < srcwordrep_doc.size(); l++) {
                for (unsigned bs = 0; bs < bsize; ++bs) {
                    if (!_p_tfc->_online_docmt) {//masking for offline document MT
                        if (sids[bs] != l) {//pad the current sentence index
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(0.f);// padding position
                        } else {
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(1.f);// padding position
                        }
                    } else {//masking for online document MT
                        if (sids[bs] > l) {//pad the current and future sentence index
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(0.f);// padding position
                        } else {
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(1.f);// padding position
                        }
                    }
                }
            }

            _word_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
            _word_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
            _word_mask.create_padding_positions_masks(_p_encoder->_self_mask._i_seq_mask, 1);
#endif
        }

        return i_sent_ctx;
    }

    dynet::Expression build_graph(dynet::ComputationGraph &cg
            , const dynet::Expression& i_src, vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids)
    {
        // get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
        dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
        dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
        dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
        dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

        //get expressions for context gating
        dynet::Expression i_Cs = dynet::parameter(cg, _p_Cs);
        dynet::Expression i_Csc = dynet::parameter(cg, _p_Csc);

        //to save the number of tokens in each sentence
        vector<unsigned int> dslen;
        dynet::Expression i_mh_att;
        if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::SENT) {//generate the sentence-level context and masks for it
            dynet::Expression i_sent_ctx = compute_sentrep_and_masks(cg, srcwordrep_doc, sids);//((num_units, ds), batch_size) here ds is number of sentences in the document

            // multi-head attention sub-layer
            i_mh_att = _context_attention_sublayer.build_graph(cg, i_src, i_sent_ctx, i_sent_ctx, _sent_mask);// ((num_units, Lx), batch_size)
        }
        else if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::WORD) {//generate the word-level context and masks for it
            dynet::Expression i_word_ctx = compute_wordrep_and_masks(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the number of tokens in the document

            // multi-head attention sub-layer
            i_mh_att = _context_attention_sublayer.build_graph(cg, i_src, i_word_ctx, i_word_ctx, _word_mask);// ((num_units, Lx), batch_size)
        }
        else if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::HIERARCHICAL) {//generate both levels of context and masks for it
            dynet::Expression i_sent_ctx = compute_sentrep_and_masks(cg, srcwordrep_doc, sids);//((num_units, ds), batch_size)
            dynet::Expression i_word_ctx = compute_wordrep_and_masks(cg, srcwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the number of tokens in the document

            // multi-head hierarchical attention sub-layer
            i_mh_att = _context_hierattention_sublayer.build_graph(cg, i_src, i_sent_ctx, i_word_ctx, i_word_ctx, _sent_mask, dslen);// ((num_units, Lx), batch_size)
        }

        // dropout to the above sub-layer
        if (_p_tfc->_use_dropout && _p_tfc->_encoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
            i_mh_att = dynet::dropout_dim(i_mh_att, 1/*col-major*/, _p_tfc->_encoder_sublayer_dropout_rate);// col-wise dropout
#else
            i_mh_att = dynet::dropout(i_mh_att, _p_tfc->_encoder_sublayer_dropout_rate);// full dropout
#endif

        // no need of residual connection
        dynet::Expression i_ctxl = /*i_src + */i_mh_att;// ((num_units, Lx), batch_size)

        // position-wise layer normalisation 1
        i_ctxl = layer_norm_colwise_3(i_ctxl, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

        // position-wise feed-forward sub-layer
        dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_ctxl);// ((num_units, Lx), batch_size)

        // w/o residual connection
        i_ctxl = /*i_ctxl + */i_ff;// ((num_units, Lx), batch_size)

        // position-wise layer normalisation 2
        i_ctxl = layer_norm_colwise_3(i_ctxl, i_ln2_g, i_ln2_b);// ((num_units, Lx), batch_size)

        /*
        cout << "For this batch: Priting contextl " << endl << endl;

        std::vector<dynet::real> contextl = as_vector(cg.get_value(i_ctxl));
        for (unsigned i = 0; i < contextl.size(); ++i){
            cout << contextl[i] << " ";
        }
        cout << endl;
        */

        //context gating
        dynet:: Expression gate_input = i_Cs * i_src + i_Csc * i_ctxl;
        dynet::Expression lambda = logistic(gate_input);
        dynet::Expression i_htilde_t = cmult(lambda, i_src) + cmult(1.f - lambda, i_ctxl);

        /*
        cout << "Printing gate op" << endl;
        std::vector<dynet::real> gateout = as_vector(cg.get_value(i_htilde_t));
        for (unsigned i = 0; i < gateout.size(); ++i){
            cout << gateout[i] << " ";
        }
        cout << endl;
        */

        return i_htilde_t;
    }
};
typedef std::shared_ptr<EncoderContext> EncoderContextPointer;
//---

//--- Decoder Context Layer
struct DecoderContext{
    explicit DecoderContext(DyNetModel* mod, TransformerConfig& tfc, Decoder* p_decoder)
            : _context_attention_sublayer(mod, tfc)
            , _context_hierattention_sublayer(mod, tfc)
            , _feed_forward_sublayer(mod, tfc)
    {
        // for layer normalisation
        _p_ln1_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
        _p_ln1_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));
        _p_ln2_g = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(1.f));
        _p_ln2_b = mod->add_parameters({tfc._num_units}, dynet::ParameterInitConst(0.f));

        // for context gating
        _p_Cs = mod->add_parameters({tfc._num_units, tfc._num_units});
        _p_Csc = mod->add_parameters({tfc._num_units, tfc._num_units});

        _p_tfc = &tfc;

        _p_decoder = p_decoder;
    }

    ~DecoderContext(){}

    //for masking the current/future sentence/s
    MaskSent _sent_mask;
    MaskSent _word_mask;

    // multi-head attention sub-layer for the sent/word-level context
    MultiHeadContextAttentionLayer _context_attention_sublayer;

    // multi-head attention sub-layer for the hierarchical context
    MultiHeadContextHierarchicalAttentionLayer _context_hierattention_sublayer;

    // position-wise feed forward sub-layer
    FeedForwardLayer _feed_forward_sublayer;

    // for layer normalisation
    dynet::Parameter _p_ln1_g, _p_ln1_b;// layer normalisation 1
    dynet::Parameter _p_ln2_g, _p_ln2_b;// layer normalisation 2

    // for context gating
    dynet::Parameter _p_Cs, _p_Csc;

    // transformer config pointer
    TransformerConfig* _p_tfc = nullptr;

    // encoder object pointer
    Decoder* _p_decoder = nullptr;

    std::pair<dynet::Expression, dynet::Expression> compute_sentrep_and_masks(dynet::ComputationGraph &cg
            , vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc, vector<unsigned int> sids)
    {
        unsigned bsize = sids.size();

        std::vector<std::vector<float>> v_seq_masks(bsize);
        vector<Expression> sent_rep, sent_att;
        for (unsigned i = 0; i < tgtwordrep_doc.size(); i++) {
            vector<std::pair<vector<dynet::real>,vector<dynet::real>>> wordrep_sent = tgtwordrep_doc[i];
            vector<Expression> word_att, word_rep;

            if (wordrep_sent.size() > 0) {
                for (unsigned j = 0; j < wordrep_sent.size(); ++j) {
                    word_att.push_back(dynet::input(cg, {_p_tfc->_num_units}, get<0>(wordrep_sent[j])));
                    word_rep.push_back(dynet::input(cg, {_p_tfc->_num_units}, get<1>(wordrep_sent[j])));
                }

                sent_att.push_back(average(word_att));
                sent_rep.push_back(average(word_rep));
            }
            else {//if the generated translation has no token
                sent_att.push_back(dynet::zeros(cg, {_p_tfc->_num_units}));
                sent_rep.push_back(dynet::zeros(cg, {_p_tfc->_num_units}));
            }
        }
        dynet::Expression i_sent_att = dynet::concatenate_to_batch(std::vector<dynet::Expression>(bsize, dynet::concatenate_cols(sent_att)));//((num_units, ds), batch_size)
        dynet::Expression i_sent_rep = dynet::concatenate_to_batch(std::vector<dynet::Expression>(bsize, dynet::concatenate_cols(sent_rep)));//((num_units, ds), batch_size)

        for (unsigned l = 0; l < tgtwordrep_doc.size(); l++){
            for (unsigned bs = 0; bs < bsize; ++bs){
                if (!_p_tfc->_online_docmt) {//masking for offline document MT
                    if (sids[bs] != l)//pad the current sentence index
                        v_seq_masks[bs].push_back(0.f);// padding position
                    else
                        v_seq_masks[bs].push_back(1.f);// padding position
                }
                else {//masking for online document MT
                    if (sids[bs] > l)//pad the current and future sentence index
                        v_seq_masks[bs].push_back(0.f);// padding position
                    else
                        v_seq_masks[bs].push_back(1.f);// padding position
                }
            }
        }

        _sent_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
        _sent_mask.create_padding_positions_masks(_p_decoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
        _sent_mask.create_padding_positions_masks(_p_decoder->_self_mask._i_seq_mask, 1);
#endif

        return make_pair(i_sent_att, i_sent_rep);
    }


    std::pair<dynet::Expression, dynet::Expression> compute_wordrep_and_masks(dynet::ComputationGraph &cg
            , vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc, vector<unsigned int> sids, vector<unsigned int>& sent_size)
    {
        unsigned bsize = sids.size();

        std::vector<std::vector<float>> v_seq_masks(bsize);
        vector<Expression> word_att, word_rep;
        for (unsigned i = 0; i < tgtwordrep_doc.size(); i++) {
            vector<std::pair<vector<dynet::real>,vector<dynet::real>>> wordrep_sent = tgtwordrep_doc[i];

            if (wordrep_sent.size() > 0) {
                for (unsigned j = 0; j < wordrep_sent.size(); ++j) {
                    word_att.push_back(dynet::input(cg, {_p_tfc->_num_units}, get<0>(wordrep_sent[j])));
                    word_rep.push_back(dynet::input(cg, {_p_tfc->_num_units}, get<1>(wordrep_sent[j])));
                }

                sent_size.push_back(wordrep_sent.size());
            }
            else {
                word_att.push_back(dynet::zeros(cg, {_p_tfc->_num_units}));
                word_rep.push_back(dynet::zeros(cg, {_p_tfc->_num_units}));

                sent_size.push_back(1);
            }
        }
        dynet::Expression i_word_att = dynet::concatenate_to_batch(std::vector<dynet::Expression>(bsize, dynet::concatenate_cols(word_att)));//((num_units, dw), batch_size)
        dynet::Expression i_word_rep = dynet::concatenate_to_batch(std::vector<dynet::Expression>(bsize, dynet::concatenate_cols(word_rep)));//((num_units, dw), batch_size)

        if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::WORD) {
            for (unsigned l = 0; l < tgtwordrep_doc.size(); l++) {
                for (unsigned bs = 0; bs < bsize; ++bs) {
                    if (!_p_tfc->_online_docmt) {//masking for offline document MT
                        if (sids[bs] != l) {//pad the current sentence index
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(0.f);// padding position
                        } else {
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(1.f);// padding position
                        }
                    } else {//masking for online document MT
                        if (sids[bs] > l) {//pad the current and future sentence index
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(0.f);// padding position
                        } else {
                            for (unsigned w = 0; w < sent_size[l]; ++w)
                                v_seq_masks[bs].push_back(1.f);// padding position
                        }
                    }
                }
            }

            _word_mask.create_seq_mask_expr(cg, v_seq_masks);
#ifdef MULTI_HEAD_ATTENTION_PARALLEL
            _word_mask.create_padding_positions_masks(_p_decoder->_self_mask._i_seq_mask, _p_tfc->_nheads);
#else
            _word_mask.create_padding_positions_masks(_p_decoder->_self_mask._i_seq_mask, 1);
#endif
        }

        return make_pair(i_word_att, i_word_rep);
    }

    dynet::Expression build_graph(dynet::ComputationGraph &cg
            , const dynet::Expression& i_tgt_ctx, const dynet::Expression& i_tgt, vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc, vector<unsigned int> sids)
    {
        // get expressions for layer normalisation, e.g., i_ln1_g, i_ln1_b, i_ln2_g, i_ln2_b
        dynet::Expression i_ln1_g = dynet::parameter(cg, _p_ln1_g);
        dynet::Expression i_ln1_b = dynet::parameter(cg, _p_ln1_b);
        dynet::Expression i_ln2_g = dynet::parameter(cg, _p_ln2_g);
        dynet::Expression i_ln2_b = dynet::parameter(cg, _p_ln2_b);

        //get expressions for context gating
        dynet::Expression i_Cs = dynet::parameter(cg, _p_Cs);
        dynet::Expression i_Csc = dynet::parameter(cg, _p_Csc);

        vector<unsigned int> dslen;
        std::pair<dynet::Expression, dynet::Expression> i_sent_ctx, i_word_ctx;
        dynet::Expression i_mh_att;
        if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::SENT) {//generate the sentence-level context and masks for it
            i_sent_ctx = compute_sentrep_and_masks(cg, tgtwordrep_doc, sids);//((num_units, ds), batch_size) here ds is number of sentences in the document

            // multi-head attention sub-layer
            i_mh_att = _context_attention_sublayer.build_graph(cg, i_tgt_ctx, get<0>(i_sent_ctx), get<1>(i_sent_ctx), _sent_mask);// ((num_units, Lx), batch_size)
        }
        else if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::WORD) {//generate the word-level context and masks for it
            i_word_ctx = compute_wordrep_and_masks(cg, tgtwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the number of tokens in the document

            // multi-head attention sub-layer
            i_mh_att = _context_attention_sublayer.build_graph(cg, i_tgt_ctx, get<0>(i_word_ctx), get<1>(i_word_ctx), _word_mask);// ((num_units, Lx), batch_size)
        }
        else if (_p_tfc->_doc_attention_type == DOC_ATTENTION_TYPE::HIERARCHICAL) {//generate both levels of context and masks for it
            i_sent_ctx = compute_sentrep_and_masks(cg, tgtwordrep_doc, sids);//((num_units, ds), batch_size)
            i_word_ctx = compute_wordrep_and_masks(cg, tgtwordrep_doc, sids, dslen);//((num_units, dw), batch_size) here dw is the number of tokens in the document

            // multi-head hierarchical attention sub-layer
            i_mh_att = _context_hierattention_sublayer.build_graph(cg, i_tgt_ctx, get<0>(i_sent_ctx), get<0>(i_word_ctx), get<1>(i_word_ctx), _sent_mask, dslen);// ((num_units, Lx), batch_size)
        }

        // dropout to the above sub-layer
        if (_p_tfc->_use_dropout && _p_tfc->_decoder_sublayer_dropout_rate > 0.f)
#ifdef USE_COLWISE_DROPOUT
            i_mh_att = dynet::dropout_dim(i_mh_att, 1/*col-major*/, _p_tfc->_decoder_sublayer_dropout_rate);// col-wise dropout
#else
            i_mh_att = dynet::dropout(i_mh_att, _p_tfc->_decoder_sublayer_dropout_rate);// full dropout
#endif

        // no need of residual connection
        dynet::Expression i_ctxl = i_mh_att;// ((num_units, Lx), batch_size)

        // position-wise layer normalisation 1
        i_ctxl = layer_norm_colwise_3(i_ctxl, i_ln1_g, i_ln1_b);// ((num_units, Lx), batch_size)

        // position-wise feed-forward sub-layer
        dynet::Expression i_ff = _feed_forward_sublayer.build_graph(cg, i_ctxl);// ((num_units, Lx), batch_size)

        // w/o residual connection
        i_ctxl = i_ff;// ((num_units, Lx), batch_size)

        // position-wise layer normalisation 2
        i_ctxl = layer_norm_colwise_3(i_ctxl, i_ln2_g, i_ln2_b);// ((num_units, Lx), batch_size)

        //context gating
        dynet:: Expression gate_input = i_Cs * i_tgt + i_Csc * i_ctxl;
        dynet::Expression lambda = logistic(gate_input);
        dynet::Expression i_htilde_t = cmult(lambda, i_tgt) + cmult(1.f - lambda, i_ctxl);

        return i_htilde_t;
    }
};
typedef std::shared_ptr<DecoderContext> DecoderContextPointer;
//---

//--- Transformer Model w/Context
struct TransformerContextModel {

public:
	explicit TransformerContextModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	explicit TransformerContextModel();

	~TransformerContextModel(){}

	// for initialisation
	void initialise(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td);

	// for training
	dynet::Expression build_graph(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents/*batched*/
		, const WordIdSentences& tsents/*batched*/
        , vector<vector<vector<dynet::real>>> srcwordrep_doc
        , vector<unsigned int> sids
		, ModelStats* pstats=nullptr
		, bool is_eval_on_dev=false);
    dynet::Expression build_graph(dynet::ComputationGraph &cg
            , const WordIdSentences& ssents/*batched*/
            , const WordIdSentences& tsents/*batched*/
            , vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc
            , vector<unsigned int> sids
            , ModelStats* pstats=nullptr
            , bool is_eval_on_dev=false);
    //for getting representations at decoding time
    std::vector<std::vector<dynet::real>> compute_source_rep(dynet::ComputationGraph &cg
            , const WordIdSentence& sent);// source representation given real sources
    std::vector<std::pair<std::vector<dynet::real>, std::vector<dynet::real>>> compute_bilingual_rep(dynet::ComputationGraph& cg, const WordIdSentence &source, unsigned length_ratio);
    dynet::Expression step_forward(dynet::ComputationGraph &cg
            , const dynet::Expression& i_src_rep
            , const WordIdSentence &partial_sent
            , vector<dynet::real>& vec_c
            , vector<dynet::real>& vec_s
            , bool log_prob
            , std::vector<dynet::Expression> &aligns
            , float sm_temp=1.f);
    // for decoding
	dynet::Expression compute_source_rep(dynet::ComputationGraph &cg
		, const WordIdSentences& ssents);// source representation given real sources
	dynet::Expression step_forward(dynet::ComputationGraph & cg
		, const dynet::Expression& i_src_rep
		, const WordIdSentence &partial_sent
		, bool log_prob
		, std::vector<dynet::Expression> &aligns
		, float sm_temp=1.f);// forward step to get softmax scores
    dynet::Expression step_forward(dynet::ComputationGraph &cg
            , const dynet::Expression& i_src_rep
            , const WordIdSentence &partial_sent
            , vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc
            , vector<unsigned int> sids
            , bool log_prob
            , std::vector<dynet::Expression> &aligns
            , float sm_temp=1.f);
    WordIdSentence greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, unsigned length_ratio=2);// greedy decoding at test time
    WordIdSentence greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, unsigned length_ratio);
    WordIdSentence greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc, vector<unsigned int> sids, unsigned length_ratio);

	dynet::ParameterCollection& get_model_parameters();
    dynet::ParameterCollection& get_context_model_parameters();
    void reset_gradient();
    void initialise_baseparams_from_file(const std::string &params_file);
    void initialise_params_from_file(const std::string &params_file);
    void save_params_to_file(const std::string &params_file);

	void set_dropout(bool is_activated = true);

	dynet::Dict& get_source_dict();
	dynet::Dict& get_target_dict();

	TransformerConfig& get_config();

protected:

    DyNetModel *_model;
    DyNetModel _base_model;
    DyNetModel _context_model;

	DyNetModelPointer _all_params;// all model parameters live in this object pointer. This object will be automatically released once unused!
    DyNetModelPointer _base_params;
    DyNetModelPointer _context_params;

	EncoderPointer _encoder;// encoder
	DecoderPointer _decoder;// decoder

    EncoderContextPointer _encoder_context;
    DecoderContextPointer _decoder_context;

	std::pair<dynet::Dict, dynet::Dict> _dicts;// pair of source and target vocabularies

	dynet::Parameter _p_Wo_bias;// bias of final linear projection layer

	TransformerConfig _tfc;// local configuration storage
};

TransformerContextModel::TransformerContextModel(){
    _all_params = nullptr;
    _base_params = nullptr;
    _context_params = nullptr;

	_encoder = nullptr;
	_decoder = nullptr;

    _encoder_context = nullptr;
    _decoder_context = nullptr;
}

TransformerContextModel::TransformerContextModel(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
: _tfc(tfc)
{
    _model = new DyNetModel();
    _all_params.reset(_model);// create new model parameter object

    _base_model = _model->add_subcollection("transformer");
    _base_params.reset(&_base_model);
    _context_model = _model->add_subcollection("context");
    _context_params.reset(&_context_model);

	_encoder.reset(new Encoder(_base_params.get(), _tfc));// create new encoder object

	_decoder.reset(new Decoder(_base_params.get(), _tfc, _encoder.get()));// create new decoder object

    if (_tfc._context_type == CONTEXT_TYPE::MONOLINGUAL)
        _encoder_context.reset(new EncoderContext(_context_params.get(), _tfc, _encoder.get()));//create new encoder context object
    else
        _decoder_context.reset(new DecoderContext(_context_params.get(), _tfc, _decoder.get()));//create new decoder context object

	// final output projection layer
	_p_Wo_bias = _base_params.get()->add_parameters({_tfc._tgt_vocab_size});// optional

	// dictionaries
	_dicts.first = sd;
	_dicts.second = td;
}

void TransformerContextModel::initialise(const TransformerConfig& tfc, dynet::Dict& sd, dynet::Dict& td)
{
    _tfc = tfc;

    _model = new DyNetModel();
    _all_params.reset(_model);// create new model parameter object

    _base_model = _model->add_subcollection("transformer");
    _base_params.reset(&_base_model);
    _context_model = _model->add_subcollection("context");
    _context_params.reset(&_context_model);

    _encoder.reset(new Encoder(_base_params.get(), _tfc));// create new encoder object

    _decoder.reset(new Decoder(_base_params.get(), _tfc, _encoder.get()));// create new decoder object

    if (_tfc._context_type == CONTEXT_TYPE::MONOLINGUAL)
        _encoder_context.reset(new EncoderContext(_context_params.get(), _tfc, _encoder.get()));//create new encoder context object
    else
        _decoder_context.reset(new DecoderContext(_context_params.get(), _tfc, _decoder.get()));//create new decoder context object

    // final output projection layer
    _p_Wo_bias = _base_params.get()->add_parameters({_tfc._tgt_vocab_size});// optional

    // dictionaries
    _dicts.first = sd;
    _dicts.second = td;
}

dynet::Expression TransformerContextModel::compute_source_rep(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents)// for decoding only
{
	// encode source
	return _encoder.get()->build_graph(cg, ssents);// ((num_units, Lx), batch_size)
}

//for getting representations
std::vector<std::vector<dynet::real>> TransformerContextModel::compute_source_rep(dynet::ComputationGraph &cg
        , const WordIdSentence& sent)
{
    std::vector<std::vector<dynet::real>> srcword_rep(sent.size());

    //encode source
    dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, WordIdSentences(1, sent));//(num_units, Lx)
    std::vector<dynet::real> i_src_td = as_vector(cg.forward(i_src_ctx));//(num_units x Lx) x 1 i.e. flattens the tensor

    auto& d = i_src_ctx.dim();
    unsigned b = d[0] * d[1];
    unsigned steps = d[0];

    //cout << d[0] << " and " << d[1] << endl;
    unsigned t;
    for (unsigned i = 0; i < b; i+=steps){//to recreate the matrix containing representations
        t = i / steps;
        std::vector<dynet::real> word_rep(i_src_td.begin() + i, i_src_td.begin() + i + steps);
        srcword_rep[t] = word_rep;
    }

    return srcword_rep;
}

//for bilingual representations from the decoder side
std::vector<std::pair<std::vector<dynet::real>, std::vector<dynet::real>>> TransformerContextModel::compute_bilingual_rep(dynet::ComputationGraph& cg, const WordIdSentence &source, unsigned length_ratio)
{
    //_tfc._is_training = false;

    std::vector<std::pair<std::vector<dynet::real>, std::vector<dynet::real>>> tgtword_rep;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);//(num_units, Lx)

    //cout << endl;
    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
        cg.checkpoint();
        vector<dynet::real> vc, vs;

        dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, vc, vs, false, aligns);//here vc and vs only have representation for the last column
        auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto pr_w = ydist[w];
        for (unsigned x = 1; x < ydist.size(); ++x) {
            if (ydist[x] > pr_w) {
                w = x;
                pr_w = ydist[w];
            }
        }

        // break potential infinite loop
        if (t > length_ratio * source.size()) {
            w = eos_sym;
            pr_w = ydist[w];
        }

        // Note: use pr_w if getting the probability of the generated sequence!

        target.push_back(w);
        t += 1;
        if (target.back() != eos_sym){//don't save representations if eos token
            tgtword_rep.push_back(make_pair(vc, vs));
        }

        //cout << (_dicts.second).convert(w) << " ";

        if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

        cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return tgtword_rep;
}

dynet::Expression TransformerContextModel::step_forward(dynet::ComputationGraph &cg
        , const dynet::Expression& i_src_rep
        , const WordIdSentence &partial_sent
        , vector<dynet::real>& vec_c
        , vector<dynet::real>& vec_s
        , bool log_prob
        , std::vector<dynet::Expression> &aligns
        , float sm_temp)
{
    // decode target
    // IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
    dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, WordIdSentences(1, partial_sent), i_src_rep, vec_c, vec_s);// the whole matrix of context representation for every words in partial_sent - which is also wasteful because we only need the representation of last comlumn?

    // only consider the prediction of last column in the matrix
    dynet::Expression i_tgt_t;
    if (partial_sent.size() == 1) i_tgt_t = i_tgt_ctx;
    else
        //i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
        i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(partial_sent.size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

    // output linear projections (w/ bias)
    dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
    dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
    dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

    // FIXME: get the alignments for visualisation
    // ToDo

    // compute softmax prediction
    if (log_prob)
        return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
    else
        return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerContextModel::step_forward(dynet::ComputationGraph &cg
	, const dynet::Expression& i_src_rep
	, const WordIdSentence &partial_sent
	, bool log_prob
	, std::vector<dynet::Expression> &aligns
	, float sm_temp)
{
	// decode target
	// IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, WordIdSentences(1, partial_sent), i_src_rep);// the whole matrix of context representation for every words in partial_sent - which is also wasteful because we only need the representation of last comlumn?

	// only consider the prediction of last column in the matrix
	dynet::Expression i_tgt_t;
	if (partial_sent.size() == 1) i_tgt_t = i_tgt_ctx;
	else 
		//i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
		i_tgt_t = dynet::pick(i_tgt_ctx, (unsigned)(partial_sent.size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

	// output linear projections (w/ bias)
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
	dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

	// FIXME: get the alignments for visualisation
	// ToDo

	// compute softmax prediction
	if (log_prob)
		return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
	else
		return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerContextModel::step_forward(dynet::ComputationGraph &cg
        , const dynet::Expression& i_src_rep
        , const WordIdSentence &partial_sent
        , vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc
        , vector<unsigned int> sids
        , bool log_prob
        , std::vector<dynet::Expression> &aligns
        , float sm_temp)
{
    // decode target
    // IMPROVEMENT: during decoding, some parts in partial_sent will be recomputed. This is wasteful, especially for beam search decoding.
    dynet::Expression i_ctx_dec, i_tgt;
    tie(i_ctx_dec, i_tgt) = _decoder.get()->build_graph_getpair(cg, WordIdSentences(1, partial_sent), i_src_rep);// the whole matrix of context representation for every words in partial_sent - which is also wasteful because we only need the representation of last comlumn?

    //combine the encoded target with the context
    dynet::Expression i_tgt_ctx_rep = _decoder_context.get()->build_graph(cg, i_ctx_dec, i_tgt, tgtwordrep_doc, sids);// ((num_units, Ly), batch_size)

    // only consider the prediction of last column in the matrix
    dynet::Expression i_tgt_t;
    if (partial_sent.size() == 1) i_tgt_t = i_tgt_ctx_rep;
    else
        //i_tgt_t = dynet::select_cols(i_tgt_ctx, {(unsigned)(partial_sent.size() - 1)});
        i_tgt_t = dynet::pick(i_tgt_ctx_rep, (unsigned)(partial_sent.size() - 1), 1);// shifted right, ((|V_T|, 1), batch_size)

    // output linear projections (w/ bias)
    dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
    dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859
    dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

    // FIXME: get the alignments for visualisation
    // ToDo

    // compute softmax prediction
    if (log_prob)
        return dynet::log_softmax(i_r_t / sm_temp);// log_softmax w/ temperature
    else
        return dynet::softmax(i_r_t / sm_temp);// softmax w/ temperature
}

dynet::Expression TransformerContextModel::build_graph(dynet::ComputationGraph &cg
	, const WordIdSentences& ssents
    , const WordIdSentences& tsents
    , vector<vector<vector<dynet::real>>> srcwordrep_doc
    , vector<unsigned int> sids
    , ModelStats* pstats
	, bool is_eval_on_dev)
{
	// encode source
    dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, pstats);// ((num_units, Lx), batch_size)

    //combine the encoded source with the context
    dynet::Expression i_src_ctx_rep = _encoder_context->build_graph(cg, i_src_ctx, srcwordrep_doc, sids);// ((num_units, Lx), batch_size)

    // decode target
	dynet::Expression i_tgt_ctx = _decoder.get()->build_graph(cg, tsents, i_src_ctx_rep);// ((num_units, Ly), batch_size)

	// get losses	
	dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
	dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING 
	// Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats)
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)
	
		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else 
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}
#else // Note: this way is much faster!
	// compute the logit and linear projections
	dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_ctx});// ((|V_T|, (Ly-1)), batch_size)

	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats) {
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// get the prediction at timestep t
		//dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
		dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// shifted right, ((|V_T|, 1), batch_size)
        /*
        cout << "Printing check_op" << endl;
        std::vector<dynet::real> check_op = as_vector(cg.get_value(i_r_t));
        for (unsigned i = 0; i < check_op.size(); ++i){
            cout << check_op[i] << " ";
        }
        cout << endl << endl;
        */

        // log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);

			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else 
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);// ((1, 1), batch_size)

		v_errors.push_back(i_err);
	}
#endif

	dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

	return i_tloss;
}

dynet::Expression TransformerContextModel::build_graph(dynet::ComputationGraph &cg
        , const WordIdSentences& ssents
        , const WordIdSentences& tsents
        , vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc
        , vector<unsigned int> sids
        , ModelStats* pstats
        , bool is_eval_on_dev)
{
    // encode source
    dynet::Expression i_src_ctx = _encoder.get()->build_graph(cg, ssents, pstats);// ((num_units, Lx), batch_size)

    // decode target and also get query needed for computing context
    dynet::Expression i_ctx_dec, i_tgt;
    tie(i_ctx_dec, i_tgt) = _decoder.get()->build_graph_getpair(cg, tsents, i_src_ctx);// ((num_units, Ly), batch_size)

    //combine the encoded target with the context
    dynet::Expression i_tgt_ctx_rep = _decoder_context.get()->build_graph(cg, i_ctx_dec, i_tgt, tgtwordrep_doc, sids);// ((num_units, Ly), batch_size)

    // get losses
    dynet::Expression i_Wo_bias = dynet::parameter(cg, _p_Wo_bias);
    dynet::Expression i_Wo_emb_tgt = dynet::transpose(_decoder.get()->get_wrd_embedding_matrix(cg));// weight tying (use the same weight with target word embedding matrix) following https://arxiv.org/abs/1608.05859

// both of the followings work well!
#ifndef USE_LINEAR_TRANSFORMATION_BROADCASTING
    // Note: can be more efficient if using direct computing for i_tgt_ctx (e.g., use affine_transform)
	std::vector<dynet::Expression> v_errors;
	unsigned tlen = _decoder.get()->_batch_tlen;
	std::vector<unsigned> next_words(tsents.size());
	for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
		for(size_t bs = 0; bs < tsents.size(); bs++){
			next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
			if (tsents[bs].size() > t && pstats)
				pstats->_words_tgt++;
				if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
			}
		}

		// compute the logit
		//dynet::Expression i_tgt_t = dynet::select_cols(i_tgt_ctx, {t});// shifted right
		dynet::Expression i_tgt_t = dynet::pick(i_tgt_ctx_rep, t, 1);// shifted right, ((|V_T|, 1), batch_size)

		// output linear projections
		dynet::Expression i_r_t = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_t});// |V_T| x 1 (with additional bias)

		// log_softmax and loss
		dynet::Expression i_err;
		if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
		{// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
			// label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
			dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
			dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
			dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
			i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
		}
		else
			i_err = dynet::pickneglogsoftmax(i_r_t, next_words);

		v_errors.push_back(i_err);
	}
#else // Note: this way is much faster!
    // compute the logit and linear projections
    dynet::Expression i_r = dynet::affine_transform({i_Wo_bias, i_Wo_emb_tgt, i_tgt_ctx_rep});// ((|V_T|, (Ly-1)), batch_size)

    std::vector<dynet::Expression> v_errors;
    unsigned tlen = _decoder.get()->_batch_tlen;
    std::vector<unsigned> next_words(tsents.size());
    for (unsigned t = 0; t < tlen - 1; ++t) {// shifted right
        for(size_t bs = 0; bs < tsents.size(); bs++){
            next_words[bs] = (tsents[bs].size() > (t + 1)) ? (unsigned)tsents[bs][t + 1] : _tfc._sm._kTGT_EOS;
            if (tsents[bs].size() > t && pstats) {
                pstats->_words_tgt++;
                if (tsents[bs][t] == _tfc._sm._kTGT_UNK) pstats->_words_tgt_unk++;
            }
        }

        // get the prediction at timestep t
        //dynet::Expression i_r_t = dynet::select_cols(i_r, {t});// shifted right, ((|V_T|, 1), batch_size)
        dynet::Expression i_r_t = dynet::pick(i_r, t, 1);// shifted right, ((|V_T|, 1), batch_size)

        // log_softmax and loss
        dynet::Expression i_err;
        if (_tfc._use_label_smoothing && !is_eval_on_dev/*only applies in training*/)
        {// w/ label smoothing (according to section 7.5.1 of http://www.deeplearningbook.org/contents/regularization.html) and https://arxiv.org/pdf/1512.00567v1.pdf.
            // label smoothing regularizes a model based on a softmax with k output values by replacing the hard 0 and 1 classification targets with targets of \epsilon / (k−1) and 1 − \epsilon, respectively!
            dynet::Expression i_log_softmax = dynet::log_softmax(i_r_t);
            dynet::Expression i_pre_loss = -dynet::pick(i_log_softmax, next_words);
            dynet::Expression i_ls_loss = -dynet::sum_elems(i_log_softmax) / (_tfc._tgt_vocab_size - 1);// or -dynet::mean_elems(i_log_softmax)
            i_err = (1.f - _tfc._label_smoothing_weight) * i_pre_loss + _tfc._label_smoothing_weight * i_ls_loss;
        }
        else
            i_err = dynet::pickneglogsoftmax(i_r_t, next_words);// ((1, 1), batch_size)

        v_errors.push_back(i_err);
    }
#endif

    dynet::Expression i_tloss = dynet::sum_batches(dynet::sum(v_errors));

    return i_tloss;
}

WordIdSentence TransformerContextModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, unsigned length_ratio)
{
    //_tfc._is_training = false;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding

    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
        cg.checkpoint();

        dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, false, aligns);
        auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto pr_w = ydist[w];
        for (unsigned x = 1; x < ydist.size(); ++x) {
            if (ydist[x] > pr_w) {
                w = x;
                pr_w = ydist[w];
            }
        }

        // break potential infinite loop
        if (t > length_ratio * source.size()) {
            w = eos_sym;
            pr_w = ydist[w];
        }

        // Note: use pr_w if getting the probability of the generated sequence!

        target.push_back(w);
        t += 1;
        if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

        cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return target;
}

WordIdSentence TransformerContextModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<vector<dynet::real>>> srcwordrep_doc, vector<unsigned int> sids, unsigned length_ratio)
{
    //_tfc._is_training = false;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding

    //combine the encoded source with the context
    dynet::Expression i_src_ctx_rep = _encoder_context->build_graph(cg, i_src_rep, srcwordrep_doc, sids);// ((num_units, Lx), 1)

    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
         cg.checkpoint();

         dynet::Expression i_ydist = this->step_forward(cg, i_src_ctx_rep, target, false, aligns);
         auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

         // find the argmax next word (greedy)
         unsigned w = 0;
         auto pr_w = ydist[w];
         for (unsigned x = 1; x < ydist.size(); ++x) {
             if (ydist[x] > pr_w) {
                 w = x;
                 pr_w = ydist[w];
             }
         }

         // break potential infinite loop
         if (t > length_ratio * source.size()) {
             w = eos_sym;
             pr_w = ydist[w];
         }

         // Note: use pr_w if getting the probability of the generated sequence!

         target.push_back(w);
         t += 1;
         if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

         cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return target;
}

WordIdSentence TransformerContextModel::greedy_decode(dynet::ComputationGraph& cg, const WordIdSentence &source, vector<vector<std::pair<vector<dynet::real>,vector<dynet::real>>>> tgtwordrep_doc,
                                                      vector<unsigned int> sids, unsigned length_ratio)
{
    //_tfc._is_training = false;

    const int& sos_sym = _tfc._sm._kTGT_SOS;
    const int& eos_sym = _tfc._sm._kTGT_EOS;

    // start of sentence
    WordIdSentence target;
    target.push_back(sos_sym);

    dynet::Expression i_src_rep = this->compute_source_rep(cg, WordIdSentences(1, source)/*pseudo batch (1)*/);// ToDo: batch decoding

    std::vector<dynet::Expression> aligns;// FIXME: unused
    unsigned t = 0;
    while (target.back() != eos_sym)
    {
        cg.checkpoint();

        //cout << "Length of partial target: " << target.size() << endl;
        dynet::Expression i_ydist = this->step_forward(cg, i_src_rep, target, tgtwordrep_doc, sids, false, aligns);
        auto ydist = dynet::as_vector(cg.incremental_forward(i_ydist));

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto pr_w = ydist[w];
        for (unsigned x = 1; x < ydist.size(); ++x) {
            if (ydist[x] > pr_w) {
                w = x;
                pr_w = ydist[w];
            }
        }

        // break potential infinite loop
        if (t > length_ratio * source.size()) {
            w = eos_sym;
            pr_w = ydist[w];
        }

        // Note: use pr_w if getting the probability of the generated sequence!

        target.push_back(w);
        t += 1;
        if (_tfc._position_encoding == 1 && t >= _tfc._max_length) break;// to prevent over-length sample in learned positional encoding

        cg.revert();
    }

    cg.clear();

    //_tfc._is_training = true;
    return target;
}

dynet::ParameterCollection& TransformerContextModel::get_model_parameters(){
	return *_all_params.get();
}

dynet::ParameterCollection& TransformerContextModel::get_context_model_parameters(){
    return *_context_params.get();
}

void TransformerContextModel::reset_gradient(){
    _model->reset_gradient();
}

void TransformerContextModel::initialise_baseparams_from_file(const std::string &params_file)
{
	//dynet::load_dynet_model(params_file, _base_params.get());// FIXME: use binary streaming instead for saving disk spaces?
    TextFileLoader loader(params_file);
    loader.populate(*_base_params.get());
}

void TransformerContextModel::initialise_params_from_file(const std::string &params_file)
{
    //dynet::load_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
    TextFileLoader loader(params_file);
    loader.populate(*_all_params.get());
}

void TransformerContextModel::save_params_to_file(const std::string &params_file)
{
	//dynet::save_dynet_model(params_file, _all_params.get());// FIXME: use binary streaming instead for saving disk spaces?
    TextFileSaver saver(params_file);
    saver.save(*_all_params.get());
}

void TransformerContextModel::set_dropout(bool is_activated){
	_tfc._use_dropout = is_activated;
}

dynet::Dict& TransformerContextModel::get_source_dict()
{
	return _dicts.first;
}
dynet::Dict& TransformerContextModel::get_target_dict()
{
	return _dicts.second;
}

TransformerConfig& TransformerContextModel::get_config(){
	return _tfc;
}

//---

}; // namespace transformer



