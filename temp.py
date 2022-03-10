        import time
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func2)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        miou = 0.
        f1_score_ = 0.
        crop_iou = 0.
        weed_iou = 0.
        recall_ = 0.
        precision_ = 0.
        f1_score_ = 0.
        final_time = 0.
        for i in range(len(test_img_dataset)):
            batch_images, nomral_img, batch_labels = next(test_iter)
            batch_labels = tf.squeeze(batch_labels, -1)
            for j in range(1):
                batch_image = tf.expand_dims(batch_images[j], 0)
                start_time = time.time()
                raw_logits = run_model(model, batch_image, False)
                raw_logits = tf.nn.sigmoid(raw_logits)
                output = run_model(model2, batch_image * raw_logits, False)
                end_time = time.time()
                if i != 0:
                    final_time += (end_time - start_time)
                    print("Inference time = {} sec.".format(final_time / i))
