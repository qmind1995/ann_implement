//
// Created by tri on 16/06/2017.
//

#ifndef ANN_IMPLEMENT_DATAREADER_H
#define ANN_IMPLEMENT_DATAREADER_H

enum { NONE, STATIC, GROWING, WINDOWING };

class DataReader
{

//private members
private:

    //data storage
    std::vector<dataEntry*> data;
    int nInputs;
    int nTargets;

    //current data set
    trainingDataSet tSet;

    //data set creation approach and total number of dataSets
    int creationApproach;
    int numTrainingSets;
    int trainingDataEndIndex;

    //creation approach variables
    double growingStepSize;			//step size - percentage of total set
    int growingLastDataIndex;		//last index added to current dataSet
    int windowingSetSize;			//initial size of set
    int windowingStepSize;			//how many entries to move window by
    int windowingStartIndex;		//window start index

//public methods
public:

    dataReader(): creationApproach(NONE), numTrainingSets(-1) {}
    ~dataReader();

    bool loadDataFile( const char* filename, int nI, int nT );
    void setCreationApproach( int approach, double param1 = -1, double param2 = -1 );
    int getNumTrainingSets();

    trainingDataSet* getTrainingDataSet();
    std::vector<dataEntry*>& getAllDataEntries();

//private methods
private:

    void createStaticDataSet();
    void createGrowingDataSet();
    void createWindowingDataSet();
    void processLine( std::string &line );
};
#endif //ANN_IMPLEMENT_DATAREADER_H
