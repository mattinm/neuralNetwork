#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPalette>

#include <string>

namespace Ui {
class MainWindow;
class Trainer;
class TrainerInfo;
}

class TrainerInfo{
public:
        std::string buildDir, cnnName,trainIdxData, trainIdxLabel, trainMosaics,testMosaics,msiLocations,outputLocation,netConfig;
        std::string excludes = "--exclude=1000000";
        unsigned int iterations;
};

class Trainer;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void disableInput();
    void enableInput();

private slots:
    void filePicker(QLineEdit* textbox, const char * filter = "*");
    void folderPicker(QLineEdit* textbox);

//    void trainCNN(const std::string &outputCNN, const std::string &termout);

    bool isNonEmpty(QLineEdit* textbox);

    void on_btnTrainingData_clicked();

    void on_btnTrainingLabel_clicked();

    void on_btnTrainMosaic_clicked();

    void on_btnTestMosaic_clicked();

    void on_btnMSILocations_clicked();

    void on_btnOutputLocation_clicked();

    void on_btnRun_clicked();

    void on_btnBuildDir_clicked();

    void on_btnNetConfig_clicked();

private:
    Ui::MainWindow *ui;
    QPalette errorPalette, standardPalette;
    bool currentlyRunning = false;
    Trainer *trainer;
};

class Trainer : public QObject {
        Q_OBJECT

public:
    Trainer(MainWindow* parent, const TrainerInfo& info);
    ~Trainer();

public slots:
    void run();

signals:
    void finished();
    void error(QString error);


private:
    MainWindow* parent;
    TrainerInfo info;
    void trainCNN(const std::string &outputCNN, const std::string &curTrainIdxData, const std::string &curTrainIdxLabel, const std::string &termout);
//    std::string buildDir, cnnName,trainIdxData, trainIdxLabel, trainMosaics,testMosaics,msiLocations,outputLocation,netConfig;
//    std::string excludes = "--exclude=1000000";
};

#endif // MAINWINDOW_H
