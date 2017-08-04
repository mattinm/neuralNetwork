#include "mainwindow.h"
#include "ui_mainwindow.h"

//Qt stuff
#include <QString>
#include <QFileDialog>
#include <QPalette>
#include <QProcess>
#include <QMessageBox>
#include <QDebug>
#include <QThread>

//C++ stuff
#include <sstream>
#include <thread>
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
}

void MainWindow::disableInput()
{
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
    info.trainIdxData = ui->txtTrainingData->text().toStdString();
    info.trainIdxLabel = ui->txtTrainingLabels->text().toStdString();
    info.cnnName = ui->txtCNNName->text().toStdString();
    info.trainMosaics = ui->txtTrainMosaic->text().toStdString();
    info.testMosaics = ui->txtTestMosaic->text().toStdString();
    info.msiLocations = ui->txtMSILocations->text().toStdString();
    info.iterations = ui->spnIterations->value();
    info.outputLocation = ui->txtOutputLocation->text().toStdString();
    info.buildDir = ui->txtBuildDir->text().toStdString();
    info.netConfig = ui->txtNetConfig->text().toStdString();

    disableInput();

    std::cout << "starting thread" << std::endl;
    //Start thread and detach it
    QThread* thr = new QThread();
    trainer = new Trainer(this,info);
    trainer->moveToThread(thr);
    connect(trainer,SIGNAL(finished()),this,SLOT(enableInput()));
//    connect(trainer, SIGNAL(error(QString)),this,SLOT(errorString(QString)));
    connect(thr, SIGNAL(started()), trainer, SLOT(run()));
    connect(trainer, SIGNAL(finished()), thr, SLOT(quit()));
    connect(trainer,SIGNAL(finished()), trainer, SLOT(deleteLater()));
    connect(thr,SIGNAL(finished()), thr, SLOT(deleteLater()));
    thr->start();
}

Trainer::Trainer(MainWindow* parent, const TrainerInfo& info)
{
    this->parent = parent;
    this->info = info;
}

Trainer::~Trainer(){}

void Trainer::run()
{
    std::cout << "in run" << std::endl;
    std::string termout = info.outputLocation + "/termout.txt";
    std::string trainMosaicResults = info.outputLocation + "/train_mosaic_results.txt";
    std::string testMosaicResults = info.outputLocation + "/test_mosaic_results.txt";
    std::string curTrainData = info.trainIdxData;
    std::string curTrainLabel = info.trainIdxLabel;
    std::stringstream blob_counts;
    std::string prevRetrainData = info.trainIdxData;
    std::string prevRetrainLabel = info.trainIdxLabel;

    for(unsigned int i = 0; i < info.iterations; i++)
    {
        std::stringstream ss;
        ss << i << "_" << info.cnnName;
        std::string curCNNName = ss.str();

        //train cnn
//        std::thread t = std::thread([=]{trainCNN(curCNNName,termout);});
        trainCNN(curCNNName,curTrainData,curTrainLabel,termout);

        //run cnn - train mosaics

        //blob count cnn - train mosaics

        //quantify results - train mosaics

        //run cnn - test mosaics

        //blob count cnn - test mosaics

        //quantify results - test mosaics
    }

    emit finished();
}

void Trainer::trainCNN(const std::string& outputCNN, const std::string& curTrainIdxData, const std::string& curTrainIdxLabel, const std::string& termout)
{
    QProcess process;
    process.setStandardOutputFile(termout.c_str());
    std::stringstream command;
    command  << info.buildDir << "/ConvNetTrainerCL_idx " << info.netConfig << " " << outputCNN << " -device=0 -trainRatio_classes=-1:2:1000000 -trainRatio_amounts=5:1:0 "
             << "-train_data=" << curTrainIdxData << " -train_label=" << curTrainIdxLabel
             << " -epochs=30 " << info.excludes;
    std::cout << command.str() << std::endl;
    connect(&process,&QProcess::errorOccurred,[=](QProcess::ProcessError error)
    {
//        QMessageBox::critical(0,tr("Fatal error"), tr("Could not start ConvNetTrainerCL_idx process."));
        qDebug() << "Error num val = " << error << '\n';
    });
    process.start(QString::fromUtf8(command.str().c_str()));
//    process.start(QString::fromUtf8("pwd"));

    std::cout << "Finsihed correctly: " << std::boolalpha << process.waitForFinished(-1) << std::endl;
}
