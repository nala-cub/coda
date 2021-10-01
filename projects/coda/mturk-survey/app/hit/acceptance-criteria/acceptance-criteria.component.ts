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
import {Component, OnInit} from '@angular/core';

@Component({
  selector : 'kln-acceptance-criteria',
  templateUrl : './acceptance-criteria.component.html',
  styles : [
    `mat-card {
      width: 100%;
    }
    div.kln-wrap-item{
      padding-left:16px;
    }
  `,
  ]
})
export class AcceptanceCriteriaComponent implements OnInit {

  constructor() {}

  ngOnInit(): void {}

  acceptable = [
    {short : `You've seen the item before`},
    {short : `Your annotations are reasonable.`},
    {short : `Your annotations are all valid.`},
    {short : `Your annotations correspond to the 'natural' state of objects.`},
  ]

  unacceptable = [
    {short : `You've never seen the item before`},
    {short : `Selecting colors for an object that is never that color`},
    {short : `Selecting every color for every object.`},
    {short : `Looking it up on Google to get image examples.`},
    {
      short : `The selected color represents a specific unnatural state of an
             object e.g. an anything covered in dirt is brown.`
    },
    {
      short : `Selected a color which represents an object being painted a
            nonstandard color (e.g. painted grass).`,
      explanation: `That's the color of paint, not grass.`
    },
  ]
}
