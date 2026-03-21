from experiments.run_step_response_nowalls import main

if __name__ == '__main__':
    main(steps=400, hold_action=0.25, out_json='experiments/step_response_nowalls_repeat1.json', out_png='experiments/plots/step_response_nowalls_repeat1.png')
    main(steps=400, hold_action=0.25, out_json='experiments/step_response_nowalls_repeat2.json', out_png='experiments/plots/step_response_nowalls_repeat2.png')
