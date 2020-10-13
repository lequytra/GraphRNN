using TensorBoardLogger, Logging, Random
using ImageMagick
using Images, FileIO


#=
Create a new logger, logging to directory log_dir
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
    return logger
end

#=
Log image using the logger.
=#
function TB_log_img(logger, tag_name, img_path, step)
    # load image
    img = load(img_path)

    # log image
    log_image(logger, tag_name, img, step = step)
end

#=
Log scalar using logger
=#
function TB_log_scalar(logger, tag_name, value, step)
    log_value(logger, tag_name, value, step = step)
end
