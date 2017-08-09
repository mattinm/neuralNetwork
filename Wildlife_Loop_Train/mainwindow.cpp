#include "mainwindow.h"
#include "ui_mainwindow.h"

//Qt stuff
#include <QString>
#include <QFileDialog>
#include <QPalette>
#include <QMessageBox>
#include <QDebug>
#include <QThread>
#include <QFile>

//C++ stuff
#include <iostream>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    errorPalette.setColor(QPalette::Base, QColor(255,204,204));
    standardPalette.setColor(QPalette::Base, Qt::white);

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
    QString foldername = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "/Users/",QFileDialog::ShowDirsOnly);
    if(foldername != NULL)
    {
        textbox->setText(foldername);
    }
}

void MainWindow::enableInput()
{
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

    ui->btnCancel->setEnabled(true);
}


MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btnTrainingData_clicked()
{
    filePicker(ui->txtTrainingData,"*.idx");
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

void MainWindow::on_btnRun_clicked()
{
    std::cout << "clicked" << std::endl;
    if(currentlyRunning)
        return;
    currentlyRunning = true;
    bool good =
        isNonEmpty(ui->txtTrainingData) &&
        isNonEmpty(ui->txtTrainingLabels) &&
        isNonEmpty(ui->txtCNNName) &&
        isNonEmpty(ui->txtTrainMosaic) &&
        isNonEmpty(ui->txtTestMosaic) &&
        isNonEmpty(ui->txtMSILocations) &&
        isNonEmpty(ui->txtOutputLocation) &&
        isNonEmpty(ui->txtBuildDir) &&
        isNonEmpty(ui->txtNetConfig)
    ;
    std::cout << "good: " << std::boolalpha << good << std::endl;
    if(!good)
    {
        currentlyRunning = false;
        return;
    }

    TrainerInfo info;
    info.trainIdxData = ui->txtTrainingData->text();
    info.trainIdxLabel = ui->txtTrainingLabels->text();
    info.cnnName = ui->txtCNNName->text().toStdString();
    info.trainMosaics = ui->txtTrainMosaic->text();
    info.testMosaics = ui->txtTestMosaic->text();
    info.msiLocations = ui->txtMSILocations->text();
    info.iterations = ui->spnIterations->value();
    info.outputLocation = ui->txtOutputLocation->text();
    info.buildDir = ui->txtBuildDir->text().toStdString();
    info.netConfig = ui->txtNetConfig->text();
    info.epochs = ui->spnEpochs->value();

    disableInput();

    std::cout << "starting thread" << std::endl;
    //Start thread and detach it
    thr = new QThread();
    trainer = new Trainer(this,info);
    trainer->moveToThread(thr);

    connect(thr, SIGNAL(started()), trainer, SLOT(run()));

    connect(trainer,SIGNAL(updateTrainingLog(QString,QString)), this, SLOT(updateTrainingLog(QString,QString)));
    connect(trainer,SIGNAL(updateCurrently(QString)),this,SLOT(updateCurrently(QString)));


    connect(trainer,SIGNAL(finished()),this,SLOT(enableInput()));
    connect(trainer, &Trainer::finished,[=](){currentlyRunning = false;});
    connect(trainer, SIGNAL(finished()), thr, SLOT(quit()));
    connect(trainer,SIGNAL(finished()), trainer, SLOT(deleteLater()));
    connect(thr,SIGNAL(finished()), thr, SLOT(deleteLater()));
    thr->start();
}

void MainWindow::updateTrainingLog(QString cnnName, QString filename)
{
    QFile file(filename);
    if(!file.exists())
    {
        qDebug() << "Error: MainWindow::updateTrainingLog - file does not exist. File '" << filename << "'\n";
    }
    else
    {
        QString line;
        if(file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            QString toAdd = "";
            toAdd += cnnName + "\n";
            QTextStream stream(&file);
            while(!stream.atEnd())
            {
                line = stream.readLine();
//                ui->txtTrainingLog->setText(ui->txtTrainingLog->toPlainText() + line + "\n");
                toAdd += line + "\n";
            }
//            QString toAddString = toAdd.string();
            ui->txtTrainingLog->append(toAdd);
        }
        file.close();
    }
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

void Trainer::run()
{
    connect(parent,&MainWindow::cancelSignal,this,[=](){cancelTrain = true;},Qt::DirectConnection);

    std::cout << "in run" << std::endl;
    QString termout = QStringLiteral("%1/termout.txt").arg(info.outputLocation);
    QString trainMosaicResults = QStringLiteral("%1/train_mosaic_results.txt").arg(info.outputLocation);
    QString testMosaicResults = QStringLiteral("%1/test_mosaic_results.txt").arg(info.outputLocation);
    QString curTrainData = info.trainIdxData;
    QString curTrainLabel = info.trainIdxLabel;
    QStringList blob_counts;
    QString prevData = info.trainIdxData;
    QString prevLabel = info.trainIdxLabel;
    QString oldCNN = info.netConfig;
    QString blobCompareFilename = QStringLiteral("%1/Blob_comparisons");
    int stride = 9;

    QString outloc, blobName;
    for(unsigned int i = 0; i < info.iterations; i++)
    {
        QString iterString = QStringLiteral("Iteration %1:").arg(i+1);
        QString curCNNName = QStringLiteral("%1/%2_%3").arg(info.buildDir.c_str()).arg(i).arg(info.cnnName.c_str());

        /********************
         * Training side
         ********************/

        //train cnn
        emit updateCurrently(iterString + " Training CNN");
        trainCNN(curCNNName,oldCNN,curTrainData,curTrainLabel,termout);
        if(cancelTrain) return;
        emit updateTrainingLog(curCNNName,termout);
        oldCNN = curCNNName;

        //run cnn - train mosaics
        emit updateCurrently(iterString + " Running over training mosaics");
        outloc = QStringLiteral("%1/%2_run_over_train").arg(info.outputLocation).arg(i);
        runCNN(curCNNName,info.trainMosaics,stride,outloc);
        if(cancelTrain) return;

        //blob count cnn - train mosaics
        emit updateCurrently(iterString + " Blob count on training mosaics");
        blobName = QStringLiteral("%1/%2_blob_count_train.csv").arg(info.outputLocation).arg(i);
        blobCount(outloc,blobName); // the outloc from runCNN is the input to this. blobName is the output of this
        if(cancelTrain) return;
        blob_counts.append(blobName);

        //do blob comparator. Doing it every iteration lets us look at intermediate results and it runs pretty fast.
        emit updateCurrently(iterString + " Comparing CNN Blob Counts to Users");
        compareBlobs(blob_counts,true_blob_counts,blobCompareFilename);
        if(cancelTrain) return;

        //quantify results - train mosaics
        emit updateCurrently(iterString + " CNN to Observer comparision for training mosaics. Getting amount misclassified BG.");
        QString loop_base  = QStringLiteral("%1/%2_loop").arg(info.outputLocation).arg(i);
        QString loop_data  = QStringLiteral("%1/%2_data.idx").arg(info.outputLocation).arg(loop_base);
        QString loop_label = QStringLiteral("%1/%2_label.idx").arg(info.outputLocation).arg(loop_base);
        compCNNObs(info.msiLocations, outloc, info.trainMosaics, loop_base, trainMosaicResults);
        if(cancelTrain) return;

        //create new IDXs for next iteration
        emit updateCurrently(iterString + " Creating new IDXs for training next iteration");
        QString retrain_base = QStringLiteral("%1/%2_retrain").arg(info.outputLocation).arg(i);
        QString retrain_data = QStringLiteral("%1_data.idx").arg(retrain_base);
        QString retrain_label = QStringLiteral("%1_label.idx").arg(retrain_base);
        combineIDXs(prevData,prevLabel,loop_data,loop_label,-1,retrain_base,termout);
        if(cancelTrain) return;

        /********************
         * Testing side
         ********************/

        //run cnn - test mosaics
        emit updateCurrently(iterString + " Running over testing mosaics");
        outloc = QStringLiteral("%1/%2_run_over_test").arg(info.outputLocation).arg(i);
        runCNN(curCNNName,info.testMosaics,stride,outloc);
        if(cancelTrain) return;

        //blob count cnn - test mosaics
        emit updateCurrently(iterString + " Blob count on testing mosaics");
        blobName = QStringLiteral("%1/%2_blob_count_test.csv").arg(info.outputLocation).arg(i);
        blobCount(outloc, blobName);
        if(cancelTrain) return;
        blob_counts.append(blobName);

        //quantify results - test mosaics
        emit updateCurrently(iterString + " CNN to Observer comparision for testing mosaics. Getting amount misclassified BG.");
        compCNNObs(info.msiLocations, outloc, testMosaicResults);
        if(cancelTrain) return;

        //do blob comparator. Doing it every iteration lets us look at intermediate results and it runs pretty fast.
        emit updateCurrently(iterString + " Comparing CNN Blob Counts to Users");
        compareBlobs(blob_counts,true_blob_counts,blobCompareFilename);
        if(cancelTrain) return;
    }

    emit finished();
}

void Trainer::trainCNN(const QString& outputCNN, const QString& oldCNN, const QString &curTrainIdxData, const QString &curTrainIdxLabel, const QString &termout)
{
    QProcess process;
    process.setStandardOutputFile(termout, QIODevice::Append);

    QString command = QStringLiteral("%1/ConvNetTrainerCL_idx %2 %3 -device=0 -trainRatio_classes=1:2:1000000 -trainRatio_amounts=5:1:0 -train_data=%4 -train_label=%5 -epochs=%6 %7 >> %8")
            .arg(info.buildDir.c_str())
            .arg(oldCNN)
            .arg(outputCNN)
            .arg(curTrainIdxData)
            .arg(curTrainIdxLabel)
            .arg(info.epochs)
            .arg(info.excludes)
            .arg(termout);

    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection); //this allows the cancel button to kill the process
//    process.start(QString::fromUtf8(command.str().c_str()));
    process.start(command);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "train Finished correctly: " << std::boolalpha << finishedCorrectly<< std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;
}

void Trainer::runCNN(const QString& cnn, const QString& imageLocation, int stride, const QString& outloc)
{
    //make folder for prediction images
    QProcess process;
    QString mkdir = QStringLiteral("mkdir %1").arg(outloc);
    process.start(mkdir);
    process.waitForFinished(-1);

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
    QProcess process;
    QString cmd = QStringLiteral("%1/BlobCounter %2/* > %3")
            .arg(info.buildDir.c_str())
            .arg(inputLocation)
            .arg(outputFilename);
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "Blob count Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
    if(!finishedCorrectly)
        cancelTrain = true;
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
    QString cmd = QStringLiteral("%1/CNNtoObserver_comparator %2 %3 %4 %5/*prediction*")
            .arg(info.buildDir.c_str())
            .arg(info.excludes)
            .arg(idxArgs)
            .arg(msi_locations)
            .arg(predImageDir);
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "CNNtoObsComp Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;

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
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "CombineIDXs Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
}

void Trainer::compareBlobs(const QStringList& blob_counts, const QString& true_blob_counts, const QString& outputBaseName)
{
    QProcess process;
    QString cmd = QStringLiteral("python %1/Blob_comparator.py %2 %3 %4")
            .arg(info.buildDir.c_str())
            .arg(outputBaseName)
            .arg(true_blob_counts)
            .arg(blob_counts.join(" "));
    connect(parent,SIGNAL(cancelSignal()),&process,SLOT(kill()),Qt::DirectConnection);
    process.start(cmd);
    bool finishedCorrectly = process.waitForFinished(-1);
    std::cout << "compareBlobs Finished correctly: " << std::boolalpha << finishedCorrectly << std::endl;
}

void MainWindow::on_btnCancel_clicked()
{
    std::cout << "cancel signal" << std::endl;
    cancelTrain();
}

void MainWindow::updateCurrently(QString status)
{
    ui->txtCurrently->setText(status);
}
