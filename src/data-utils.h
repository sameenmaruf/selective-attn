#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "dynet/dict.h"

using namespace std;
using namespace dynet;

WordIdCorpus read_corpus(const string &filename
	, dynet::Dict* sd, dynet::Dict* td
	, bool cid=true/*corpus id, 1:train;0:otherwise*/
	, unsigned slen=0, bool r2l_target=false
	, bool swap=false);

WordIdSentences read_corpus(const string &filename
	, dynet::Dict* d
	, bool cid=true/*corpus id, 1:train;0:otherwise*/
	, unsigned slen=0, bool r2l_target=false);

//The functions for document-level corpus by Sameen Maruf
SentCorpus read_doccorpus(const string &filename, dynet::Dict* sd, dynet::Dict* td);
DocCorpus Read_DocCorpus(SentCorpus &corpus);
void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers);

SrcCorpus read_srcdoccorpus(const string &filename, dynet::Dict* sd, dynet::Dict* td);
SrcDocCorpus read_srcdoccorpus(SrcCorpus &corpus);
void Read_Numbered_Sentence(const std::string& line, std::vector<int>* s, Dict* sd, vector<int> &identifiers);

WordIdCorpus read_corpus(const string &filename
	, dynet::Dict* sd, dynet::Dict* td
	, bool cid
	, unsigned slen, bool r2l_target
	, bool swap)
{
	bool use_joint_vocab = false;
	if (sd == td) use_joint_vocab = true;

	int kSRC_SOS = sd->convert("<s>");
	int kSRC_EOS = sd->convert("</s>");
	int kTGT_SOS = td->convert("<s>");
	int kTGT_EOS = td->convert("</s>");

	ifstream in(filename);
	assert(in);

	WordIdCorpus corpus;

	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	unsigned int max_src_len = 0, max_tgt_len = 0;
	while (getline(in, line)) {
		WordIdSentence source, target;

		if (!swap)
			read_sentence_pair(line, source, *sd, target, *td);
		else read_sentence_pair(line, source, *td, target, *sd);

		// reverse the target if required
		if (r2l_target) 
			std::reverse(target.begin() + 1/*BOS*/, target.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (source.size() > slen || target.size() > slen)
				continue;// ignore this sentence
		}

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			stringstream ss;
			ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			assert(ss.str().c_str());

			abort();
		}

		if (source.size() < 3 || target.size() < 3){ // ignore empty sentences, e.g., <s> </s>
			continue;
		}

		corpus.push_back(WordIdSentencePair(source, target));

		max_src_len = std::max(max_src_len, (unsigned int)source.size());
		max_tgt_len = std::max(max_tgt_len, (unsigned int)target.size());

		stoks += source.size();
		ttoks += target.size();

		++lc;
	}

	// print stats
	if (cid){
		if (!use_joint_vocab)
			cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " & " << td->size() << " types" << endl;
		else 
			cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " joint s & t types" << endl;
	}
	else 
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << endl;

	return corpus;
}

WordIdSentences read_corpus(const string &filename
	, dynet::Dict* d
	, bool cid
	, unsigned slen, bool r2l)
{
	int SOS = d->convert("<s>");
	int EOS = d->convert("</s>");

	ifstream in(filename);
	assert(in);

	WordIdSentences corpus;

	string line;
	int lc = 0, toks = 0;
	unsigned int max_len = 0;
	while (getline(in, line)) {
		WordIdSentence sent = read_sentence(line, *d);

		// reverse the target if required
		if (r2l) 
			std::reverse(sent.begin() + 1/*BOS*/, sent.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (sent.size() > slen)
				continue;// ignore this sentence
		}

		if (sent.front() != SOS && sent.back() != EOS) {
			stringstream ss;
			ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			assert(ss.str().c_str());

			abort();
		}

		if (sent.size() < 3){ // ignore empty sentences, e.g., <s> </s>
			continue;
		}

		corpus.push_back(sent);

		max_len = std::max(max_len, (unsigned int)sent.size());

		toks += sent.size();

		++lc;
	}

	// print stats
	if (cid)
		cerr << lc << " lines, " << toks << " tokens, " << "max length: " << max_len << ", " << d->size() << " types" << endl;
	else 
		cerr << lc << " lines, " << toks << " tokens, " << "max length: " << max_len << endl;

	return corpus;
}
//--------------------------------------------------------------------------------
//function to read the corpus with docid's. Output is a bilingual parallel corpus with docid
SentCorpus read_doccorpus(const string &filename
		, dynet::Dict* sd, dynet::Dict* td)
{
    bool use_joint_vocab = false;
    if (sd == td) use_joint_vocab = true;

    int kSRC_SOS = sd->convert("<s>");
    int kSRC_EOS = sd->convert("</s>");
    int kTGT_SOS = td->convert("<s>");
    int kTGT_EOS = td->convert("</s>");

    ifstream in(filename);
	assert(in);
	SentCorpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
    unsigned int max_src_len = 0, max_tgt_len = 0;
    vector<int> identifiers({ -1 });
	while (getline(in, line)) {
		WordIdSentence source, target;
		Read_Numbered_Sentence_Pair(line, &source, sd, &target, td, identifiers);

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
            (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            stringstream ss;
            ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            assert(ss.str().c_str());

            abort();
        }

		corpus.push_back(SentencePairID(source, target, identifiers[0]));

        max_src_len = std::max(max_src_len, (unsigned int)source.size());
        max_tgt_len = std::max(max_tgt_len, (unsigned int)target.size());

        stoks += source.size();
		ttoks += target.size();

        ++lc;
    }

    if (!use_joint_vocab)
        cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " & " << td->size() << " types" << endl;
    else
        cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " joint s & t types" << endl;

    return corpus;
}

//function to convert the bilingual parallel corpus with docid to document-level corpus
DocCorpus read_doccorpus(SentCorpus &corpus)
{
	//for loop to create a document level corpus
	vector<WordIdSentencePair> document;
	DocCorpus doccorpus;
	int docid = 0, prev_docid = 1;
	for (unsigned int index = 0; index < corpus.size(); ++index)
	{
		docid = get<2>(corpus.at(index));
		if (index > 0)
			prev_docid = get<2>(corpus.at(index - 1));
		else
			prev_docid = docid;
		if (docid == prev_docid)
			document.push_back(WordIdSentencePair(get<0>(corpus.at(index)),get<1>(corpus.at(index))));
		else{
			doccorpus.push_back(document);
			document.clear();
			document.push_back(WordIdSentencePair(get<0>(corpus.at(index)),get<1>(corpus.at(index))));
		}
	}
	doccorpus.push_back(document);	//push the last document read onto the doccorpus
	cerr << doccorpus.size() << " # of documents\n";

	return doccorpus;
}

void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::vector<int>* v = s;

    if (in) {
        identifiers.clear();
        while (in >> word) {
            if (!in || word.empty()) break;
            if (word == sep) break;
            identifiers.push_back(atoi(word.c_str()));
        }
    }

    while(in) {
        in >> word;
        if (!in) break;
        if (word == sep) { d = td; v = t; continue; }
        v->push_back(d->convert(word));
    }
}

//---
//function to read the source corpus with docid's. Output is a monolingual corpus with docid
SrcCorpus read_srcdoccorpus(const string &filename, dynet::Dict* sd, dynet::Dict* td)
{
    bool use_joint_vocab = false;
    if (sd == td) use_joint_vocab = true;

    int kSRC_SOS = sd->convert("<s>");
    int kSRC_EOS = sd->convert("</s>");
    //int kTGT_SOS = td->convert("<s>");
    //int kTGT_EOS = td->convert("</s>");

    ifstream in(filename);
    assert(in);
    SrcCorpus corpus;
    string line;
    int lc = 0, stoks = 0;
    vector<int> identifiers({ -1 });
    while (getline(in, line)) {
        WordIdSentence source;
        Read_Numbered_Sentence(line, &source, sd, identifiers);

        corpus.push_back(SentenceID(source, identifiers[0]));

        if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }

        stoks += source.size();
        ++lc;
    }

    if (!use_joint_vocab)
        cerr << lc << " lines, " << stoks << " tokens (s), " << sd->size() << " & " << td->size() << " types" << endl;
    else
        cerr << lc << " lines, " << stoks << " tokens (s), " << sd->size() << " joint s & t types" << endl;

    return corpus;
}

//function to read the source corpus with docid's. Output is a monolingual corpus with docid
SrcDocCorpus read_srcdoccorpus(SrcCorpus &corpus)
{
    //for loop to create a document level corpus
    WordIdSentences document;
    SrcDocCorpus doccorpus;
    int docid = 0, prev_docid = 1;
    for (unsigned int index = 0; index < corpus.size(); ++index)
    {
        docid = get<1>(corpus.at(index));
        if (index > 0)
            prev_docid = get<1>(corpus.at(index - 1));
        else
            prev_docid = docid;
        if (docid == prev_docid)
            document.push_back(get<0>(corpus.at(index)));
        else{
            doccorpus.push_back(document);
            document.clear();
            document.push_back(get<0>(corpus.at(index)));
        }
    }
    doccorpus.push_back(document);	//push the last document read onto the doccorpus
    cerr << doccorpus.size() << " # of documents\n";

    return doccorpus;
}

void Read_Numbered_Sentence(const std::string& line, std::vector<int>* s, Dict* sd, vector<int> &identifiers) {
    std::istringstream in(line);
    std::string word;
    std::vector<int>* v = s;
    std::string sep = "|||";
    if (in) {
        identifiers.clear();
        while (in >> word) {
            if (!in || word.empty()) break;
            if (word == sep) break;
            identifiers.push_back(atoi(word.c_str()));
        }
    }

    while(in) {
        in >> word;
        if (!in || word.empty()) break;
        v->push_back(sd->convert(word));
    }
}
