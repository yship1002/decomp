// Copyright (C) 2025, Gurobi Optimization, LLC
// All Rights Reserved
#ifndef _CPP_ENV_H_
#define _CPP_ENV_H_


class GRBEnv
{
  private:

    GRBenv*  env;
    GRBenv** envP;

// only for gurobi_c++.h const GRBEnv& operator=(const GRBEnv &xenv);

    GRBEnv(GRBenv *Cenv);

  public:

    friend class GRBModel;
    friend class GRBBatch;

    GRBEnv(const bool empty = false);
    GRBEnv(const char* logfilename);
    GRBEnv(const std::string& logfilename);
    GRBEnv(const GRBEnv& xenv) : env(xenv.env),
                                 envP(xenv.envP) {}

    ~GRBEnv();
    void start();
    void message(const std::string& msg);
    int get(GRB_IntParam param) const;
    double get(GRB_DoubleParam param) const;
    std::string get(GRB_StringParam param) const;
    void set(GRB_IntParam param, int newvalue);
    void set(GRB_DoubleParam param, double newvalue);
    void set(GRB_StringParam param, const std::string& newvalue);
    void set(GRB_StringParam param, const char* newvalue);
    void set(const std::string& paramname, const std::string& newvalue);
    void set(const std::string& paramname, const char* newvalue);
    void getParamInfo(GRB_DoubleParam param, double* valP,
                      double* minP, double* maxP, double* defP);
    void getParamInfo(GRB_IntParam param, int* valP, int* minP,
                      int* maxP, int* defP);
    void getParamInfo(GRB_StringParam param, std::string& value,
                      std::string& defvalue);
    void resetParams();
    void writeParams(const std::string& paramfile);
    void readParams(const std::string& paramfile);
    const std::string getErrorMsg() const;
    int getWLSTokenLifespan() const;
};
#endif
