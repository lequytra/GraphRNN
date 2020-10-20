using TensorBoardLogger, Logging, Random
using ImageMagick
using Images, FileIO


#=
Create a new logger, logging to directory log_dir. One logger can log multiple
type of data. We can provide "tag name" for each data to differentiate between them.

Params:
    log_dir: String: The directory to store the log files
    delete_old: Bool: if true then the old logs will be removed else
                the new log will append the old logs.

Return:
    logger: TBLogger: a logger to log data.
=#
function  TB_set_up(log_dir, delete_old=true)
    # first, we clean the old log
    if delete_old
        try
            run(string("rm -r", log_dir))
        catch
            # do nothing
        end
    end

    # create new TB logger
    logger = TBLogger(log_dir, tb_append)
    @info "Log to Tensorboard at $log_dir"
    return logger
end




#=
Log image using the logger.

Params:
    logger: TBLogger
    tag_name: String: name of the data being logged
    img_path: String: the directory to the image being logged.
    step: Int: current step in the logging.
=#
function TB_log_img(logger, tag_name, img_path, step)
    # load image
    img = load(img_path)

    # log image
    log_image(logger, tag_name, img, step = step)
end




#=
Log scalar using logger
Params:
    logger: TBLogger
    tag_name: String: name of the data being logged
    img_path: String: the directory to the image being logged.
    step: Int: current step in the logging.
=#
function TB_log_scalar(logger, tag_name, value, step)
    log_value(logger, tag_name, value, step = step)
end
