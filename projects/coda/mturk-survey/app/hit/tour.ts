/**
 * Copyright 2021 Cory Paik
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import {TourConfig} from '../intro/introjs.service';

export const tourConfig: TourConfig = {
  showProgress: false,
  scrollToElement: false,
  steps: [
    {
      intro: 'Hello! This is an introduction to how to do Object Color  ' +
          'Labeling. Click Next.'
    },
    {
      element: '#concept-card',
      intro: 'This is the object that you will be analyzing for the current ' +
          'task. To complete the task, you should be familiar with the color ' +
          'of this object.'
    },
    {
      element: '#colortable',
      intro: `For each of the listed colors, use the sliders below to
          indicate how frequently the object is that color. Use a scale of 0-5
          in your ratings where 0 corresponds to almost never and 5 corresponds
          to almost always.`
    },
    {
      element: '#colortable',
      intro: `By default, all of the sliders are set to 0 (almost never).
          Sliding all the way to the right indicates a score of 5/5`
    },
    {
      element: '#submitbox',
      intro: `Here you'll find the actions for progressing through tasks.`
    },
    {
      element: '#clear-btn',
      intro: `Use this to clear out all current color ratings in the table.`
    },
    {
      element: '#submit-btn',
      intro: `This button will be enabled once your color ratings are valid. An
           empty rating is invalid.`,
      position: 'left'
    },
    {
      element: '#more-info',
      intro: `You can access more instructions for the task here. Detailed
          instructions summarizes this demo, and to repeat the demo press "Show
          Task Demo". `,
    },
  ]
};
