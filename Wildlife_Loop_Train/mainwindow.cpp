#include "mainwindow.h"
#include "ui_mainwindow.h"

//Qt stuff
#include <QString>
#include <QPalette>
#include <QMessageBox>
#include <QDebug>
#include <QThread>
#include <QFile>
#include <QFileInfo>
#include <QFileInfoList>
#include <QFileDialog>
#include <QByteArray>
#include <QDir>
#include <QTime>


//C++ stuff
#include <iostream>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    errorPalette.setColor(QPalette::Base, QColor(255,204,204)); // need input
    standardPalette.setColor(QPalette::Base, Qt::white);
    missingPalette.setColor(QPalette::Base, QColor(255,255,204)); // file/folder doesn't exist

    ui->lgdNoInput->setPalette(errorPalette);
    ui->lgdFileNotExist->setPalette(missingPalette);

    blobModel = new QStandardItemModel;
    ui->tblBlobResults->setSelectionBehavior(QAbstractItemView::SelectItems);
    ui->tblBlobResults->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tblBlobResults->setModel(blobModel);
}

void MainWindow::setNumTrials(int num_trials)
{
    ui->spnNumTrials->setValue(num_trials);
}

void MainWindow::setStartingNet(const QString &filepath)
{
    ui->txtNetConfig->setText(filepath);
}
void MainWindow::setTrainData(const QString &filepath)
{
    ui->txtTrainingData->setText(filepath);
}
void MainWindow::setTrainLabel(const QString &filepath)
{
    ui->txtTrainingLabels->setText(filepath);
}
void MainWindow::setOutputCNN(const QString &filepath)
{
    ui->txtCNNName->setText(filepath);
}
void MainWindow::setTrainMosaic(const QString &folderpath)
{
    ui->txtTrainMosaic->setText(folderpath);
}
void MainWindow::setTestMosaic(const QString &folderpath)
{
    ui->txtTestMosaic->setText(folderpath);
}
void MainWindow::setMSILocations(const QString &filepath)
{
    ui->txtMSILocations->setText(filepath);
}
void MainWindow::setTrueBlobCount(const QString &filepath)
{
    ui->txtTrueBlobCounts->setText(filepath);
}
void MainWindow::setIterationsRetrain(int iterations)
{
    ui->spnIterations->setValue(iterations);
}
void MainWindow::setEpochs(int epochs)
{
    ui->spnEpochs->setValue(epochs);
}
void MainWindow::setOutputLocation(const QString &folderpath)
{
    ui->txtOutputLocation->setText(folderpath);
}
void MainWindow::setBuildDir(const QString &folderpath)
{
    ui->txtBuildDir->setText(folderpath);
}


void MainWindow::filePicker(QLineEdit* textbox, const char * filter)
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open File"),"~/",tr(filter));
    if(filename != NULL)
    {
        textbox->setText(filename);
    }
}

void MainWindow::folderPicker(QLineEdit* textbox)
{
    QString foldername = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "~/",QFileDialog::ShowDirsOnly);
    if(foldername != NULL)
    {
        textbox->setText(foldername);
    }
}

bool MainWindow::fileCheckEmptyAndExists(QLineEdit* textbox)
{
    bool nonEmpty = isNonEmpty(textbox), exists;
    if(nonEmpty)
        exists = fileExists(textbox);
    return nonEmpty && exists;
}

bool MainWindow::folderCheckEmptyAndExists(QLineEdit* textbox)
{
    bool nonEmpty = isNonEmpty(textbox), exists;
    if(nonEmpty)
        exists = folderExists(textbox);
    return nonEmpty && exists;
}

bool MainWindow::fileExists(QLineEdit* textbox)
{
    QString path = textbox->text();
    QFileInfo file(path);
    bool good = file.exists() && file.isFile();
    if(good)
        textbox->setPalette(standardPalette);
    else
        textbox->setPalette(missingPalette);
    return good;
}

bool MainWindow::folderExists(QLineEdit* textbox)
{
    QString path = textbox->text();
    QFileInfo folder(path);
    bool good =  folder.exists() && folder.isDir();
    if(good)
        textbox->setPalette(standardPalette);
    else
        textbox->setPalette(missingPalette);
    return good;
}

void MainWindow::enableInput()
{
//    std::cout << "enable input" << std::endl;
    ui->spnEpochs->setEnabled(true);
    ui->btnTrainingData->setEnabled(true);
    ui->btnTrainingLabel->setEnabled(true);
    ui->btnTrainMosaic->setEnabled(true);
    ui->btnTestMosaic->setEnabled(true);
    ui->btnMSILocations->setEnabled(true);
    ui->btnOutputLocation->setEnabled(true);
    ui->btnBuildDir->setEnabled(true);
    ui->btnNetConfig->setEnabled(true);
    ui->txtTrainingData->setEnabled(true);
    ui->txtTrainingLabels->setEnabled(true);
    ui->txtCNNName->setEnabled(true);
    ui->txtTrainMosaic->setEnabled(true);
    ui->txtTestMosaic->setEnabled(true);
    ui->txtMSILocations->setEnabled(true);
    ui->spnIterations->setEnabled(true);
    ui->txtOutputLocation->setEnabled(true);
    ui->txtBuildDir->setEnabled(true);
    ui->txtNetConfig->setEnabled(true);
    ui->btnRun->setEnabled(true);
    ui->txtTrueBlobCounts->setEnabled(true);
    ui->btnTrueBlobCounts->setEnabled(true);
    ui->spnNumTrials->setEnabled(true);

    ui->btnCancel->setEnabled(false);
}

void MainWindow::disableInput()
{
    ui->spnEpochs->setEnabled(false);
    ui->btnTrainingData->setEnabled(false);
    ui->btnTrainingLabel->setEnabled(false);
    ui->btnTrainMosaic->setEnabled(false);
    ui->btnTestMosaic->setEnabled(false);
    ui->btnMSILocations->setEnabled(false);
    ui->btnOutputLocation->setEnabled(false);
    ui->btnBuildDir->setEnabled(false);
    ui->btnNetConfig->setEnabled(false);
    ui->txtTrainingData->setEnabled(false);
    ui->txtTrainingLabels->setEnabled(false);
    ui->txtCNNName->setEnabled(false);
    ui->txtTrainMosaic->setEnabled(false);
    ui->txtTestMosaic->setEnabled(false);
    ui->txtMSILocations->setEnabled(false);
    ui->spnIterations->setEnabled(false);
    ui->txtOutputLocation->setEnabled(false);
    ui->txtBuildDir->setEnabled(false);
    ui->txtNetConfig->setEnabled(false);
    ui->btnRun->setEnabled(false);
    ui->txtTrueBlobCounts->setEnabled(false);
    ui->btnTrueBlobCounts->setEnabled(false);
    ui->spnNumTrials->setEnabled(false);

    ui->btnCancel->setEnabled(true);
}


MainWindow::~MainWindow()
{
    delete ui;
    delete blobModel;
}

void MainWindow::on_btnTrainingData_clicked()
{
    filePicker(ui->txtTrainingData,"*.idx");
}

void MainWindow::on_btnTrueBlobCounts_clicked()
{
    filePicker(ui->txtTrueBlobCounts,"*.csv");
}

void MainWindow::on_btnTrainingLabel_clicked()
{
    filePicker(ui->txtTrainingLabels,"*.idx");
}

void MainWindow::on_btnTrainMosaic_clicked()
{
    folderPicker(ui->txtTrainMosaic);
}

void MainWindow::on_btnTestMosaic_clicked()
{
    folderPicker(ui->txtTestMosaic);
}

void MainWindow::on_btnMSILocations_clicked()
{
    filePicker(ui->txtMSILocations);
}

void MainWindow::on_btnOutputLocation_clicked()
{
    folderPicker(ui->txtOutputLocation);
}

void MainWindow::on_btnBuildDir_clicked()
{
    folderPicker(ui->txtBuildDir);
}

void MainWindow::on_btnNetConfig_clicked()
{
    filePicker(ui->txtNetConfig);
}

bool MainWindow::isNonEmpty(QLineEdit* textbox)
{
    bool nonEmpty = !textbox->text().isEmpty();
    if(nonEmpty)
        textbox->setPalette(standardPalette);
    else //is empty
        textbox->setPalette(errorPalette);
    return nonEmpty;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    cancelTrain();
    event->accept();
}

bool MainWindow::run()
{
//    std::cout << "clicked" << std::endl;
    if(!currentlyRunning)
    {
        currentlyRunning = true;
        bool good =
            fileCheckEmptyAndExists(ui->txtTrainingData) * //&& this is a non-shortcircuit AND
            fileCheckEmptyAndExists(ui->txtTrainingLabels) * //&&
            isNonEmpty(ui->txtCNNName) * //&&
            folderCheckEmptyAndExists(ui->txtTrainMosaic) * //&&
            folderCheckEmptyAndExists(ui->txtTestMosaic) * //&&
            fileCheckEmptyAndExists(ui->txtMSILocations) * //&&
            folderCheckEmptyAndExists(ui->txtOutputLocation) * //&&
            folderCheckEmptyAndExists(ui->txtBuildDir) * //&&
            fileCheckEmptyAndExists(ui->txtNetConfig) * //&&
            fileCheckEmptyAndExists(ui->txtTrueBlobCounts)
        ;
    //    std::cout << "non empty good: " << std::boolalpha << good << std::endl;
        if(!good)
        {
            currentlyRunning = false;
            return false;
        }
        disableInput();

        info.trainIdxData = ui->txtTrainingData->text();
        info.trainIdxLabel = ui->txtTrainingLabels->text();
        info.trainMosaics = ui->txtTrainMosaic->text();
        info.testMosaics = ui->txtTestMosaic->text();
        info.msiLocations = ui->txtMSILocations->text();
        info.iterations = ui->spnIterations->value();
        info.buildDir = ui->txtBuildDir->text().toStdString();
        info.netConfig = ui->txtNetConfig->text();
        info.epochs = ui->spnEpochs->value();
        info.trueBlobCounts = ui->txtTrueBlobCounts->text();
        info.rootOutputLocation = ui->txtOutputLocation->text();

        numTrials = ui->spnNumTrials->value();
        curTrial = 0;

        ui->txtTrainingLog->setText("");
    }

    QString dir = QStringLiteral("%1/trial%2").arg(ui->txtOutputLocation->text()).arg(curTrial);
    info.outputLocation = dir;
    QDir().mkdir(dir);

    info.cnnName = QStringLiteral("trial%1_%2").arg(curTrial).arg(ui->txtCNNName->text());


//    std::cout << "starting thread" << std::endl;
    //Start thread and detach it
    thr = new QThread();
    trainer = new Trainer(this,info);
    trainer->moveToThread(thr);

    connect(thr, SIGNAL(started()), trainer, SLOT(run()));

    connect(trainer,SIGNAL(updateTrainingLog(QString,QString)), this, SLOT(updateTrainingLog(QString,QString)));
    connect(trainer,SIGNAL(updateBlobTable(QString)),this,SLOT(updateBlobTable(QString)));
    connect(trainer,SIGNAL(updateCurrently(QString)),this,SLOT(updateCurrently(QString)));

    connect(trainer,SIGNAL(finished()),this,SLOT(endRun()));
//    connect(trainer,SIGNAL(finished()),this,SLOT(enableInput()));
//    connect(trainer, &Trainer::finished,[=](){currentlyRunning = false;});
    connect(trainer, SIGNAL(finished()), thr, SLOT(quit()));
//    connect(trainer,SIGNAL(finished()), trainer, SLOT(deleteLater()));
    connect(thr,SIGNAL(finished()), thr, SLOT(deleteLater()));
    thr->start();
    return true;
}

void MainWindow::endRun()
{
    curTrial++;
    if(!trainer->finishedWell())
        currentlyRunning = false;
//    delete thr;
    delete trainer;
    if(curTrial < numTrials && currentlyRunning)
    {
        updateTrainingLog("","____________________________________________");
        run();
    }
    else
    {
        enableInput();
        currentlyRunning = false;
        csvFiles.clear();
    }
}

void MainWindow::on_btnRun_clicked()
{
    run();
}

void MainWindow::updateTrainingLog(QString cnnName, QString info)
{
    ui->txtTrainingLog->append(cnnName + "\n" + info + "\n");
}

void MainWindow::cancelTrain()
{
    emit cancelSignal();
    currentlyRunning = false;
}

Trainer::Trainer(MainWindow* parent, const TrainerInfo& info)
{
    this->parent = parent;
    this->info = info;
}

Trainer::~Trainer(){}

bool Trainer::finishedWell() const
{
    return finished_well;
}

void Trainer::run()
{
    finished_well = false;
    connect(parent,&MainWindow::cancelSignal,this,[=](){cancelTrain = true;},Qt::DirectConnection);

//    std::cout << "in run" << std::endl;
    QString termout = QStringLiteral("%1/termout.txt").arg(info.outputLocation);
    QString trainMosaicResults = QStringLiteral("%1/train_mosaic_results.txt").arg(info.outputLocation);
    QString testMosaicResults = QStringLiteral("%1/test_mosaic_results.txt").arg(info.outputLocation);
    QString curTrainData = info.trainIdxData;
    QString curTrainLabel = info.trainIdxLabel;
    QStringList blob_counts;
    QString oldCNN = info.netConfig;
    QString blobCompareFilename = QStringLiteral("%1/Blob_comparisons").arg(info.outputLocation);
    QString blobCompareAgg = QStringLiteral("%1_aggregate.csv").arg(blobCompareFilename);
    QString blobCompareIndiv = QStringLiteral("%1_individuals.csv").arg(blobCompareFilename);
    int stride = 9;

    qsrand((uint)QTime::currentTime().msec());
    //make soft links to all the training mosaic files and use them for runCNN
    QString shortTrainMosaics = QStringLiteral("%1/shortTrainMosaics").arg(info.outputLocation);
    QDir().mkdir(shortTrainMosaics);
    QDir tmDir(info.trainMosaics);
    QStringList filter;
    filter << "*.png" << "*.jpg" << "*.jpeg";
    tmDir.setNameFilters(filter);
    QFileInfoList files = tmDir.entryInfoList();
    for(QFileInfo file : files)
        QFile::link(file.absoluteFilePath(),QStringLiteral("%1/%2").arg(shortTrainMosaics).arg(file.fileName()));

    QString outloc, blobName;
    for(unsigned int i = 0; i < info.iterations; i++)
    {
        QString iterString = QStringLiteral("Iteration %1:").arg(i+1);
        QString curCNNName = QStringLiteral("%1/%2_%3").arg(info.outputLocation).arg(i).arg(info.cnnName);

        /********************
         * Training side
         ********************/


        //train cnn
        emit updateCurrently(iterString + " Training CNN");
        trainCNN(curCNNName,oldCNN,curTrainData,curTrainLabel,termout);
        if(cancelTrain) break;

        //run cnn - train mosaics
        emit updateCurrently(iterString + " Running over training mosaics");
        outloc = QStringLiteral("%1/%2_run_over_train").arg(info.outputLocation).arg(i);
        runCNN(curCNNName,shortTrainMosaics,stride,outloc);
        if(cancelTrain) break;

        //blob count cnn - train mosaics
        emit updateCurrently(iterString + " Blob count on training mosaics");
        blobName = QStringLiteral("%1/%2_blob_count_train.csv").arg(info.outputLocation).arg(i);
        blobCount(outloc,blobName); // the outloc from runCNN is the input to this. blobName is the output of this
        if(cancelTrain) break;
        blob_counts.append(blobName);

        //do blob comparator. Doing it every iteration lets us look at intermediate results and it runs pretty fast.
        emit updateCurrently(iterString + " Comparing CNN Blob Counts to Users");
        compareBlobs(blob_counts,info.trueBlobCounts,blobCompareFilename);
        if(cancelTrain) break;
        emit updateBlobTable(blobCompareAgg);

        //Determine images to be trained on next iteration
        emit updateCurrently(iterString + " Determining training mosaics for next iteration");
        QDir shortTrainDir(shortTrainMosaics);
        shortTrainDir.setNameFilters(filter);
        for(QFileInfo file : shortTrainDir.entryInfoList())
            QFile(file.absoluteFilePath()).remove();
        QFileInfoList nextMosaics;
        determineNextMosaics(blobCompareIndiv, i, info.trainMosaics, nextMosaics);
        for(QFileInfo file : nextMosaics)
            QFile::link(file.absoluteFilePath(),QStringLiteral("%1/%2").arg(shortTrainMosaics).arg(file.fileName()));

        //quantify results - train mosaics
        emit updateCurrently(iterString + " CNN to Observer comparision for training mosaics. Getting amount misclassified BG.");
        QString loop_base  = QStringLiteral("%1/%2_loop").arg(info.outputLocation).arg(i);
        QString loop_data  = QStringLiteral("%1_data.idx").arg(loop_base);
        QString loop_label = QStringLiteral("%1_label.idx").arg(loop_base);
        compCNNObs(info.msiLocations, outloc, info.trainMosaics, loop_base, trainMosaicResults);
        if(cancelTrain) break;

        //create new IDXs for next iteration
        emit updateCurrently(iterString + " Creating new IDXs for training next iteration");
        QString retrain_base = QStringLiteral("%1/%2_retrain").arg(info.outputLocation).arg(i);
        QString retrain_data = QStringLiteral("%1_data.idx").arg(retrain_base);
        QString retrain_label = QStringLiteral("%1_label.idx").arg(retrain_base);
        combineIDXs(curTrainData,curTrainLabel,loop_data,loop_label,-1,retrain_base,termout);
        if(cancelTrain) break;


        /********************
         * Testing side
         ********************/

        //run cnn - test mosaics
        emit updateCurrently(iterString + " Running over testing mosaics");
        outloc = QStringLiteral("%1/%2_run_over_test").arg(info.outputLocation).arg(i);
        runCNN(curCNNName,info.testMosaics,stride,outloc);
        if(cancelTrain) break;

        //blob count cnn - test mosaics
        emit updateCurrently(iterString + " Blob count on testing mosaics");
        blobName = QStringLiteral("%1/%2_blob_count_test.csv").arg(info.outputLocation).arg(i);
        blobCount(outloc, blobName);
        if(cancelTrain) break;
        blob_counts.append(blobName);

        //do blob comparator. Doing it every iteration lets us look at intermediate results and it runs pretty fast.
        emit updateCurrently(iterString + " Comparing CNN Blob Counts to Users");
        compareBlobs(blob_counts,info.trueBlobCounts,blobCompareFilename);
        if(cancelTrain) break;
        emit updateBlobTable(blobCompareAgg);

        //quantify results - test mosaics
        emit updateCurrently(iterString + " CNN to Observer comparision for testing mosaics. Getting amount misclassified BG.");
        compCNNObs(info.msiLocations, outloc, testMosaicResults);
        if(cancelTrain) break;


        /********************
         * Setup for next iteration
         ********************/
        curTrainData = retrain_data;
        curTrainLabel = retrain_label;
        oldCNN = curCNNName;
    }
    if(cancelTrain)
        emit updateCurrently("Cancelled");
    else
    {
        emit updateCurrently("Done");
        finished_well = true;
    }
    emit finished();
}

void Trainer::trainCNN(const QString& outputCNN, const QString& oldCNN, const QString &curTrainIdxData, const QString &curTrainIdxLabel, const QString &termout)
{
    QProcess process;

    QString command = QStringLiteral("%1/ConvNetTrainerCL_idx %2 %3 -device=0 -trainRatio_classes=-1:2:1000000 -trainRatio_amounts=1:1:0 -train_data=%4 -train_label=%5 -epochs=%6 %7")
            .arg(info.buildDir.c_str())
            .arg(oldCNN)
            .arg(outputCNN)
            .arg(curTrainIdxData)
            .arg(curTrainIdxLabel)
            .arg(info.epochs)
            .arg(info.excludes);

    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection); //this allows the cancel button to kill the process
    process.start(command);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "train Finished correctly: " << std::boolalpha << finishedCorrectly<< std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;

    //get output
    QString out = process.readAllStandardOutput();

    //send to termout
    QFile term(termout);
    term.open(QIODevice::Append | QIODevice::WriteOnly);
    term.write(out.toUtf8());
    term.close();

    //give training log cnn name w/out path and best epoch stats
    emit updateTrainingLog(outputCNN.mid(outputCNN.lastIndexOf('/')+1),getBestEpoch(out));
}

void Trainer::runCNN(const QString& cnn, const QString& imageLocation, int stride, const QString& outloc)
{
    //make folder for prediction images
    QProcess process;
//    QString mkdir = QStringLiteral("mkdir %1").arg(outloc);
//    process.start(mkdir);
//    process.waitForFinished(-1);
    QDir().mkdir(outloc);

    //run cnn over mosaics
    QString runcmd = QStringLiteral("%1/ConvNetFullImageDriverParallelCL %2 %3 stride=%4 -rgb -excludeDevice=1 -excludeDevice=2 -outloc=%5")
            .arg(info.buildDir.c_str())
            .arg(cnn)
            .arg(imageLocation)
            .arg(stride)
            .arg(outloc);
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
    process.start(runcmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "run Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;
}

void Trainer::blobCount(const QString& inputLocation, const QString& outputFilename)
{
//    std::cout << inputLocation.toStdString() << " : " << outputFilename.toStdString() << std::endl;
    QProcess process;
    process.setStandardOutputFile(outputFilename, QIODevice::Append);
    QDir predictions(inputLocation);
    QStringList paths, filter, smallPaths;
    filter << "*.png" << "*.jpg" << "*.jpeg";
    predictions.setNameFilters(filter);
    QFileInfoList files = predictions.entryInfoList();
    for(QFileInfo file : files)
        paths.append(file.absoluteFilePath());
    int pathsSizeMinus1 = paths.size() - 1;
    for(int i = 0; i < paths.size(); i++)
    {
        smallPaths.append(paths[i]);
        if((i % 800 == 0 && i != 0) || i == pathsSizeMinus1)
        {
            QString cmd = QStringLiteral("%1/BlobCounter")
                    .arg(info.buildDir.c_str());
                    //.arg(smallPaths.join(' '));
            connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
        //    std::cout << cmd.toStdString() << std::endl;
            if(cancelTrain) break;
            process.start(cmd, smallPaths);
            bool finishedCorrectly = process.waitForFinished(-1);
            std::cout << "Blob count Finished correctly: " << std::boolalpha << finishedCorrectly << ":" << paths.size() << std::endl;
            if(!finishedCorrectly)
            {
                cancelTrain = true;
                break;
            }
            smallPaths.clear();
        }
    }
}

void Trainer::compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& outputFilename, bool appendOutput)
{
    _compCNNObs(msi_locations,predImageDir,outputFilename,appendOutput,"");
}

void Trainer::compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& origImageDir, const QString& idx_base_name, const QString& outputFilename, bool appendOutput)
{
    QString idxArgs = QStringLiteral("--idx_size=18 --original_image_folder=%1 --idx_name=%2")
            .arg(origImageDir)
            .arg(idx_base_name);
    _compCNNObs(msi_locations,predImageDir,outputFilename, appendOutput,idxArgs);
}

void Trainer::_compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& outputFilename, bool appendOutput, const QString& idxArgs)
{
    QProcess process;
    if(appendOutput)
        process.setStandardOutputFile(outputFilename, QIODevice::Append);
    else
        process.setStandardOutputFile(outputFilename);
//    QStringList filter;
//    filter << "*prediction*";
//    QDir predictions(predImageDir);
//    predictions.setNameFilters(filter);
//    QStringList paths;
//    QFileInfoList files = predictions.entryInfoList();
//    for(QFileInfo file : files)
//        paths.append(file.absoluteFilePath());
    QString cmd = QStringLiteral("%1/CNNtoObserver_comparator %2 %3 %4 %5")
            .arg(info.buildDir.c_str())
            .arg(info.excludes)
            .arg(idxArgs)
            .arg(msi_locations)
            .arg(predImageDir);
//            .arg(paths.join(' '));
//    std::cout << cmd.toStdString() << std::endl;
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "CNNtoObsComp Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;

}

void Trainer::combineIDXs(const QString& prev_data, const QString& prev_label, const QString& loop_data, const QString& loop_label, double percentFromOld, const QString& baseOutputName, const QString& terminalOutputFile, bool appendOutput)
{
    QProcess process;
    if(appendOutput)
        process.setStandardOutputFile(terminalOutputFile, QIODevice::Append);
    else
        process.setStandardOutputFile(terminalOutputFile);
    QString cmd = QStringLiteral("%1/CombineIDX_Thesis %2 %3 %4 %5 %6 %7 %8")
            .arg(info.buildDir.c_str())
            .arg(prev_data)
            .arg(prev_label)
            .arg(loop_data)
            .arg(loop_label)
            .arg(percentFromOld)
            .arg(baseOutputName)
            .arg(info.excludes);
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
//    std::cout << "CombineIDX cmd: " << cmd.toStdString() << std::endl << std::endl;
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "CombineIDXs Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;
}

void Trainer::compareBlobs(const QStringList& blob_counts, const QString& true_blob_counts, const QString& outputBaseName)
{
    QProcess process;
    QString cmd = QStringLiteral("/usr/bin/python %1/Blob_comparator.py %2 %3 %4")
            .arg(info.buildDir.c_str())
            .arg(outputBaseName)
            .arg(true_blob_counts)
            .arg(blob_counts.join(" "));
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
//    std::cout << cmd.toStdString() << std::endl;
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "compareBlobs Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;
}

void Trainer::determineNextMosaics(const QString &blob_indiv_path, const int iteration, const QString &trainMosaicFolder, QFileInfoList &dest)
{
    dest.clear();
    std::cout << "determineNextMosaics" << std::endl;
    QFile file(blob_indiv_path);
    if(!file.open(QIODevice::ReadOnly))
    {
       std::cout << file.errorString().toStdString() << std::endl;
       exit(1);
    }
    QStringList msisToKeep;
    while(!file.atEnd())
    {
        QString line = file.readLine();
        if(line.contains(QStringLiteral("%1_blob_count_train.csv").arg(iteration)))
        {
            QStringList parts = line.split(',');
            //cnn, msi, calc white, actual white, err white, abs err white, ...

            std::cout << "msi: " << parts[1].toInt() << " abs err: " << parts[5].toInt() << std::endl;
            if(parts[5].toInt() != 0) // if there is error keep, if there is no error keep at 30% chance
                msisToKeep.append(parts[1]);

        }
    }


    QDir tmDir(trainMosaicFolder);
    std::cout << "Size of tmf " << tmDir.entryInfoList(QDir::NoDotAndDotDot | QDir::Files).size() << std::endl;
    for(QFileInfo info : tmDir.entryInfoList(QDir::NoDotAndDotDot | QDir::Files))
    {
        bool found = false;
        for(int i = 0; i < msisToKeep.size(); i++)
        {
            if(info.fileName().contains(msisToKeep[i]))
            {
                std::cout << "Keeping " << msisToKeep[i].toStdString() << " because of errors" << std::endl;
                dest.append(info);
                msisToKeep.removeAt(i);
                found = true;
                break;
            }
        }
        if(!found && qrand() % 10 < 3)
        {
            dest.append(info);
            std::cout << "Keeping " << info.fileName().toStdString() << " because of random" << std::endl;
        }
    }

    std::cout << "Using " << dest.size() << " training images for next iteration" << std::endl;
}

void MainWindow::on_btnCancel_clicked()
{
//    std::cout << "cancel signal" << std::endl;
    cancelTrain();
}

void MainWindow::updateCurrently(QString status)
{
    status = QStringLiteral("Trial %1 : %2").arg(curTrial).arg(status);
    ui->txtCurrently->setText(status);
}

QString Trainer::getBestEpoch(QString output)
{
//    std::cout << "initial:" << output.toStdString() << std::endl;
    int startOfEpochs = output.indexOf("Epoch:");
    output.remove(0,startOfEpochs);
//    std::cout << "startepochs: " << startOfEpochs << " leftover " << output.toStdString() << std::endl;
    QStringList epochs = output.split("Epoch:",QString::SkipEmptyParts);
    double best = 0;
    double bestIndex = -1;
    for(int i = 0; i < epochs.size(); i++)
    {
        int firstPercentSign = epochs[i].indexOf('%');
        int startOfPercentage = epochs[i].lastIndexOf(' ',firstPercentSign) + 1; // space before percentage + 1
        bool goodConversion;
        double current = epochs[i].midRef(startOfPercentage,firstPercentSign-startOfPercentage).toDouble(&goodConversion);
//        std::cout << "Good conversion: " << std::boolalpha << goodConversion << std::endl;
        if(current > best)
        {
            best = current;
            bestIndex = i;
        }
    }
    if(bestIndex != -1)
        return "Epoch: " + epochs[bestIndex].left(epochs[bestIndex].indexOf("Changed learning"));
    return "";
}

void MainWindow::updateBlobTable(QString filename)
{
    if(!csvFiles.contains(filename))
        csvFiles.append(filename);
    int lineindex = 0;
    QVector<QStringList> trains, test;
//    qDebug() << csvFiles.size();
    for(int c = 0; c < csvFiles.size(); c++)
    {
//        qDebug() << c << " : " << csvFiles[c];
        QFile file(csvFiles[c]);
        if(file.open(QIODevice::ReadOnly))
        {
            int lineNum = 0;
            QTextStream in(&file);
            while(!in.atEnd())
            {
                QString line = in.readLine();
                if(lineindex == 0) //read in header as is
                {
                    QStringList linetoken = line.split(",",QString::SkipEmptyParts);
                    for(int j = 0; j < linetoken.size(); j++)
                    {
                        QString val = linetoken.at(j);

                        QStandardItem *item = new QStandardItem(val);
                        blobModel->setItem(lineindex, j, item);
                    }
                    lineindex++;
                }
                else if(lineNum == 0); // skip the headers all files but the first
                else // sort rest into train and test
                {
                    if(line.contains("blob_count_train.csv"))
                        trains.append(line.split(",",QString::SkipEmptyParts));
                    else
                        test.append(line.split(",",QString::SkipEmptyParts));
                }
                lineNum++;
            }
            file.close();
        }
    }

    for(int i = 0; i < trains.size(); i++)
    {
        for(int j = 0; j < trains[i].size(); j++)
        {
            QString val = trains[i].at(j);
            int lastSlash = val.lastIndexOf('/');
            if(j == 0)
                val = val.mid(val.lastIndexOf('/',lastSlash-1)+1);
            QStandardItem *item = new QStandardItem(val);
            blobModel->setItem(lineindex, j, item);
        }
        lineindex++;
    }
    for(int i = 0; i < test.size(); i++)
    {
        for(int j = 0; j < test[i].size(); j++)
        {
            QString val = test[i].at(j);
            int lastSlash = val.lastIndexOf('/');
            if(j == 0)
                val = val.mid(val.lastIndexOf('/',lastSlash-1)+1);
            QStandardItem *item = new QStandardItem(val);
            blobModel->setItem(lineindex, j, item);
        }
        lineindex++;
    }
}



