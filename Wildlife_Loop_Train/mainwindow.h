#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPalette>
#include <QCloseEvent>
#include <QProcess>
#include <QThread>
#include <QStringList>

#include <string>

namespace Ui {
class MainWindow;
class Trainer;
class TrainerInfo;
}

class TrainerInfo{
public:
        std::string buildDir, cnnName;
        QString netConfig, excludes = "--exclude=1000000", msiLocations, trainMosaics, testMosaics, outputLocation, trainIdxData, trainIdxLabel;
        unsigned int iterations, epochs;
};

class Trainer;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

signals:
    void closingWindow();
    void cancelSignal();

public slots:
    void disableInput();
    void enableInput();
    void cancelTrain();
    void closeEvent(QCloseEvent* event);

private slots:
    void updateTrainingLog(QString cnnName, QString filename);
    void updateCurrently(QString status);


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

    void on_btnCancel_clicked();

private:
    Ui::MainWindow *ui;
    QPalette errorPalette, standardPalette;
    bool currentlyRunning = false;
    Trainer *trainer;
    QThread *thr;
};

class Trainer : public QObject {
        Q_OBJECT

public:
    Trainer(MainWindow* parent, const TrainerInfo& info);
    ~Trainer();

public slots:
    void run();

signals:
    void endProcess();
    void finished();
    void error(QString error);
    void updateTrainingLog(QString cnnName, QString filename);
    void updateCurrently(QString status);


private:
    MainWindow *parent;
    TrainerInfo info;
    bool cancelTrain = false;
    void trainCNN(const QString &outputCNN, const QString &oldCNN, const QString &curTrainIdxData, const QString &curTrainIdxLabel, const QString &termout);
    void runCNN(const QString& cnn, const QString& imageLocation, int stride, const QString& outloc);
    void blobCount(const QString& inputLocation, const QString& outputFilename);
    void compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& outputFilename, bool appendOutput = true);
    void compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& origImageDir, const QString& idx_base_name, const QString& outputFilename, bool appendOutput = true);
    void _compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& outputFilename, bool appendOutput, const QString& idxArgs);
    void combineIDXs(const QString& prev_data, const QString& prev_label, const QString& loop_data, const QString& loop_label, double percentFromOld, const QString& baseOutputName, const QString& terminalOutputFile, bool appendOutput = true);
    void compareBlobs(const QStringList &blob_counts, const QString& true_blob_counts, const QString &outputBaseName);

};

#endif // MAINWINDOW_H
