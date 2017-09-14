#include "mainwindow.h"
#include <QApplication>
#include <iostream>

QString usage()
{
    QString out = "Mac: ./Wildlife_Loop_Train/Contents/MacOS/Wildlife_Loop_Train [options]\n";
    out += "Options: For args needing a value, can be --arg=value or --arg value\n";
    out += "   --no_gui                         No gui. Calls run with default args or cmd line args if specified. Requires --output_cnn arg.\n";
    out += "   --num_trials int                 Sets the number of trials to run to int.\n";
    out += "   --net filepath                   Sets the base network to filepath. Can be a NetConfig or a trained CNN\n";
    out += "   --train_data filepath.idx        Sets the base training IDX data file to filepath\n";
    out += "   --train_label filepath.idx       Sets the base training IDX label file to filepath\n";
    out += "   --output_cnn string              Sets the base name for the resulting CNNs to string\n";
    out += "   --train_mosaic folderpath        Sets the training mosaic folder to folderpath\n";
    out += "   --test_mosaic folderpath         Sets the testing mosaic folder to folderpath\n";
    out += "   --msi_locations filepath.bin     Sets the MSI Locations binary file to filepath\n";
    out += "   --true_blob_count filepath.csv   Sets the true blob count CSV file to filepath\n";
    out += "   --iterations_retrain int         Sets the number of retrain iterations to int\n";
    out += "   --epochs int                     Sets the number of training epochs per iteration to int\n";
    out += "   --output_location folderpath     Sets the output location for the resulting files to folderpath. Folderpath should exist.\n";
    out += "   --build_dir folderpath           Sets the location of the build of neuralNetwork to be used.\n";
    out += "\n";
    out += "   --help                           Displays this usage\n";

    return out;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    bool gui = true;
    for(int a = 1; a < argc; a++)
    {
        QString arg(argv[a]);
        if(arg.contains("--help"))
        {
            std::cout << usage().toStdString() << '\n';
            return 0;
        }
        else if(arg.contains("--no_gui"))
            gui = false;
        else if(arg.contains("--num_trials"))
        {
            if(arg.contains("="))
                w.setNumTrials(arg.mid(arg.indexOf('=')+1).toInt());
            else
                w.setNumTrials(atoi(argv[++a]));
        }
        else if(arg.contains("--net"))
        {
            if(arg.contains('='))
                w.setStartingNet(arg.mid(arg.indexOf('=')+1));
            else
                w.setStartingNet(QString(argv[++a]));
        }
        else if(arg.contains("--train_data"))
        {
            if(arg.contains("="))
                w.setTrainData(arg.mid(arg.indexOf('=')+1));
            else
                w.setTrainData(QString(argv[++a]));
        }
        else if(arg.contains("--train_label"))
        {
            if(arg.contains("="))
                w.setTrainLabel(arg.mid(arg.indexOf('=')+1));
            else
                w.setTrainLabel(QString(argv[++a]));
        }
        else if(arg.contains("--output_cnn"))
        {
            if(arg.contains("="))
                w.setOutputCNN(arg.mid(arg.indexOf('=')+1));
            else
                w.setOutputCNN(QString(argv[++a]));
        }
        else if(arg.contains("--train_mosaic"))
        {
            if(arg.contains("="))
                w.setTrainMosaic(arg.mid(arg.indexOf('=')+1));
            else
                w.setTrainMosaic(QString(argv[++a]));
        }
        else if(arg.contains("--test_mosaic"))
        {
            if(arg.contains("="))
                w.setTestMosaic(arg.mid(arg.indexOf('=')+1));
            else
                w.setTestMosaic(QString(argv[++a]));
        }
        else if(arg.contains("--msi_locations"))
        {
            if(arg.contains("="))
                w.setMSILocations(arg.mid(arg.indexOf('=')+1));
            else
                w.setMSILocations(QString(argv[++a]));
        }
        else if(arg.contains("--true_blob_count"))
        {
            if(arg.contains("="))
                w.setTrueBlobCount(arg.mid(arg.indexOf('=')+1));
            else
                w.setTrueBlobCount(QString(argv[++a]));
        }
        else if(arg.contains("--iterations_retrain"))
        {
            if(arg.contains("="))
                w.setIterationsRetrain(arg.mid(arg.indexOf('=')+1).toInt());
            else
                w.setIterationsRetrain(atoi(argv[++a]));
        }
        else if(arg.contains("--epochs"))
        {
            if(arg.contains("="))
                w.setEpochs(arg.mid(arg.indexOf('=')+1).toInt());
            else
                w.setEpochs(atoi(argv[++a]));
        }
        else if(arg.contains("--output_location"))
        {
            if(arg.contains("="))
                w.setOutputLocation(arg.mid(arg.indexOf('=')+1));
            else
                w.setOutputLocation(QString(argv[++a]));
        }
        else if(arg.contains("--build_dir"))
        {
            if(arg.contains("="))
                w.setBuildDir(arg.mid(arg.indexOf('=')+1));
            else
                w.setBuildDir(QString(argv[++a]));
        }
        else
        {
            std::cerr << "Unknown arg '" << arg.toStdString() << "'\n";
            return 1;
        }


    }

    if(gui)
        w.show();
    else
    {
        bool startedCorrectly = w.run();
        if(startedCorrectly)
            std::cout << "Run started correctly\n";
        else
            std::cout << "Run was NOT started correctly\n" << usage().toStdString() << '\n';
    }

    return a.exec();
}
